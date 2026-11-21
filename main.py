import os
import io
import re
import json
import base64
import logging

from urllib.parse import quote_plus, urlparse
from typing import Optional, List, Dict, Any

import requests
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# =========================================================
# 1. CONFIGURATION
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pricedekho")

APP_NAME = "PriceDekho AI"
APP_VERSION = "4.1.0"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

MAX_IMAGE_SIZE = 10 * 1024 * 1024

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="AI Product Identification and Smart Marketplace Matching"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================================================
# 2. DATA MODELS
# =========================================================

class Product(BaseModel):
    brand: str = ""
    model: str = ""
    category: str = ""
    variant: str = ""
    confidence: float = 0.0
    evidence: str = ""


class ProductLinkRequest(BaseModel):
    brand: str
    model: str
    variant: str = ""


class DirectLinkRequest(BaseModel):
    brand: str
    model: str
    variant: str = ""
    marketplace: str
    product_url: str
    candidate_title: str = ""


class CandidateMatchRequest(BaseModel):
    brand: str
    model: str
    variant: str = ""
    marketplace: str
    product_url: str
    candidate_title: str


# =========================================================
# 3. PYDANTIC COMPATIBILITY
# =========================================================

def model_to_dict(model):

    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


# =========================================================
# 4. MARKETPLACE CONFIGURATION
# =========================================================

MARKETPLACE_CONFIG = [
    {
        "name": "Amazon",
        "search": "https://www.amazon.in/s?k={query}"
    },
    {
        "name": "Flipkart",
        "search": "https://www.flipkart.com/search?q={query}"
    },
    {
        "name": "Meesho",
        "search": "https://www.meesho.com/search?q={query}"
    },
    {
        "name": "Google Shopping",
        "search": "https://www.google.com/search?tbm=shop&q={query}"
    },
    {
        "name": "Myntra",
        "search": "https://www.myntra.com/search?query={query}"
    },
    {
        "name": "AJIO",
        "search": "https://www.ajio.com/search/?text={query}"
    },
    {
        "name": "Tata CLiQ",
        "search": "https://www.tatacliq.com/search/?searchCategory=all&text={query}"
    }
]


ALLOWED_DOMAINS = {
    "Amazon": ["amazon.in"],
    "Flipkart": ["flipkart.com"],
    "Meesho": ["meesho.com"],
    "Google Shopping": ["google.com"],
    "Myntra": ["myntra.com"],
    "AJIO": ["ajio.com"],
    "Tata CLiQ": ["tatacliq.com"]
}


# =========================================================
# 5. PRODUCT IDENTIFICATION USING GEMINI
# =========================================================

def identify_product(image_bytes: bytes, mime_type: str) -> Product:

    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key is not configured."
        )

    prompt = """
You are an accurate product identification AI.

Analyze the uploaded product image.

Identify:
1. Brand name
2. Exact model name
3. Product category
4. Variant, color, size or capacity if visible
5. Confidence from 0 to 1
6. Visual evidence

Rules:
- Do not guess model names.
- Do not invent specifications.
- If model is unclear, return an empty model.
- Only use visible evidence.
- Return valid JSON only.

JSON format:
{
  "brand": "",
  "model": "",
  "category": "",
  "variant": "",
  "confidence": 0.0,
  "evidence": ""
}
"""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + GEMINI_MODEL
        + ":generateContent?key="
        + GEMINI_API_KEY
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(
                                image_bytes
                            ).decode("utf-8")
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json"
        }
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        result = response.json()

        text = (
            result["candidates"][0]
            ["content"]["parts"][0]["text"]
        )

        data = json.loads(text)

        confidence = float(data.get("confidence", 0))

        return Product(
            brand=str(data.get("brand", "")).strip(),
            model=str(data.get("model", "")).strip(),
            category=str(data.get("category", "")).strip(),
            variant=str(data.get("variant", "")).strip(),
            confidence=max(0.0, min(1.0, confidence)),
            evidence=str(data.get("evidence", "")).strip()
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception("Gemini identification failed")

        raise HTTPException(
            status_code=502,
            detail="Product identification failed. Please try again."
        )


# =========================================================
# 6. PRODUCT QUERY AND VALIDATION
# =========================================================

def is_exact_product(product: Product) -> bool:

    return (
        bool(product.brand.strip())
        and bool(product.model.strip())
        and product.confidence >= 0.75
    )


def make_product_query(product: Product) -> str:

    parts = [
        product.brand,
        product.model,
        product.variant
    ]

    return " ".join(
        str(part).strip()
        for part in parts
        if str(part).strip()
    )


def normalize_text(value: str) -> str:

    value = str(value or "").lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    return " ".join(value.split())


def get_product_tokens(value: str) -> set:

    ignored_words = {
        "the", "and", "with", "for",
        "new", "original", "product"
    }

    return {
        word
        for word in normalize_text(value).split()
        if word not in ignored_words
    }


# =========================================================
# 7. SMART PRODUCT TITLE MATCHING
# =========================================================

def compare_product_title(
    product: Product,
    candidate_title: str
) -> Dict[str, Any]:

    expected_brand = get_product_tokens(product.brand)
    expected_model = get_product_tokens(product.model)
    expected_variant = get_product_tokens(product.variant)

    candidate_tokens = get_product_tokens(candidate_title)

    if not expected_brand or not expected_model:
        return {
            "score": 0,
            "brand_match": False,
            "model_match": False,
            "variant_match": False,
            "possible_match": False
        }

    brand_match = expected_brand.issubset(candidate_tokens)

    model_match = expected_model.issubset(candidate_tokens)

    variant_match = (
        not expected_variant
        or expected_variant.issubset(candidate_tokens)
    )

    matched_tokens = (
        expected_brand.intersection(candidate_tokens)
        | expected_model.intersection(candidate_tokens)
        | expected_variant.intersection(candidate_tokens)
    )

    total_tokens = (
        expected_brand
        | expected_model
        | expected_variant
    )

    score = (
        len(matched_tokens) / len(total_tokens) * 100
        if total_tokens
        else 0
    )

    possible_match = (
        brand_match
        and model_match
        and variant_match
        and score >= 80
    )

    return {
        "score": round(score, 2),
        "brand_match": brand_match,
        "model_match": model_match,
        "variant_match": variant_match,
        "possible_match": possible_match
    }


# =========================================================
# 8. MARKETPLACE URL VALIDATION
# =========================================================

def validate_direct_url(name: str, url: str) -> bool:

    try:
        parsed = urlparse(str(url).strip())

        if parsed.scheme != "https":
            return False

        if not parsed.hostname:
            return False

        if parsed.username or parsed.password:
            return False

        host = parsed.hostname.lower()

        domains = ALLOWED_DOMAINS.get(name, [])

        return any(
            host == domain or host.endswith("." + domain)
            for domain in domains
        )

    except Exception:
        return False


# =========================================================
# 9. AMAZON ASIN DETECTION
# =========================================================

def extract_amazon_asin(url: str) -> Optional[str]:

    try:
        parsed = urlparse(url)

        if not validate_direct_url("Amazon", url):
            return None

        patterns = [
            r"/dp/([A-Z0-9]{10})",
            r"/gp/product/([A-Z0-9]{10})",
            r"/gp/aw/d/([A-Z0-9]{10})"
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                parsed.path,
                re.IGNORECASE
            )

            if match:
                return match.group(1).upper()

        return None

    except Exception:
        return None


def create_amazon_direct_url(asin: str) -> Optional[str]:

    asin = str(asin).strip().upper()

    if not re.fullmatch(r"[A-Z0-9]{10}", asin):
        return None

    return "https://www.amazon.in/dp/" + asin


# =========================================================
# 10. MARKETPLACE SEARCH LINKS
# =========================================================

def create_marketplace_links(
    product: Product,
    verified_direct_urls: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:

    verified_direct_urls = verified_direct_urls or {}

    query = quote_plus(make_product_query(product))

    links = []

    for marketplace in MARKETPLACE_CONFIG:

        name = marketplace["name"]

        direct_url = verified_direct_urls.get(name, "")

        # Only use a direct URL when it has been supplied
        # by a trusted source and independently confirmed.
        # This backend does not automatically confirm identity.

        if (
            direct_url
            and validate_direct_url(name, direct_url)
        ):

            links.append({
                "name": name,
                "url": direct_url,
                "link_type": "direct_candidate",
                "exact_match": False,
                "match_status": "identity_not_verified"
            })

        else:

            search_url = marketplace["search"].format(
                query=query
            )

            links.append({
                "name": name,
                "url": search_url,
                "link_type": "search",
                "exact_match": False,
                "match_status": "search_only"
            })

    return links


# =========================================================
# 11. MARKETPLACE STATUS
# =========================================================

def get_marketplace_status():

    result = []

    for item in MARKETPLACE_CONFIG:

        result.append({
            "name": item["name"],
            "status": "search_link_available",
            "direct_link_status": "verified_product_data_required",
            "price_status": "not_connected"
        })

    return result


# =========================================================
# 12. PRODUCT OFFERS
# =========================================================

def find_exact_offers(product: Product) -> List[Dict[str, Any]]:

    # No official marketplace product feeds are connected yet.
    # Never invent prices or offers.

    return []


def find_cheapest_offer(
    offers: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:

    valid_offers = [
        offer for offer in offers
        if isinstance(offer.get("price"), (int, float))
        and offer["price"] > 0
        and offer.get("exact_match") is True
    ]

    if not valid_offers:
        return None

    return min(
        valid_offers,
        key=lambda item: item["price"]
    )


# =========================================================
# 13. RESPONSE BUILDER
# =========================================================

def build_product_response(
    product: Product,
    verified_direct_urls: Optional[Dict[str, str]] = None
):

    exact_match = is_exact_product(product)

    marketplace_links = (
        create_marketplace_links(
            product,
            verified_direct_urls
        )
        if exact_match
        else []
    )

    offers = (
        find_exact_offers(product)
        if exact_match
        else []
    )

    cheapest_offer = find_cheapest_offer(offers)

    return {
        "success": True,
        "exact_match": exact_match,
        "message": (
            "Product identified. Marketplace links are ready."
            if exact_match
            else "Product could not be identified with enough confidence."
        ),
        "product": model_to_dict(product),
        "offers": offers,
        "cheapest_offer": cheapest_offer,
        "marketplace_links": marketplace_links,
        "marketplaces": get_marketplace_status()
    }


# =========================================================
# 14. HOME AND HEALTH APIs
# =========================================================

@app.get("/")
def home():

    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "status": "running"
    }


@app.get("/api/v1/health")
def health():

    return {
        "status": "ok",
        "app": APP_NAME,
        "version": APP_VERSION
    }


# =========================================================
# 15. MARKETPLACE API
# =========================================================

@app.get("/api/v1/marketplaces")
def marketplaces():

    return {
        "success": True,
        "marketplaces": get_marketplace_status()
    }


# =========================================================
# 16. PRODUCT SEARCH LINK API
# =========================================================

@app.get("/api/v1/product-links")
def product_links(
    brand: str,
    model: str,
    variant: str = ""
):

    product = Product(
        brand=brand,
        model=model,
        variant=variant,
        confidence=1.0
    )

    return {
        "success": True,
        "product": model_to_dict(product),
        "marketplace_links": create_marketplace_links(product),
        "marketplaces": get_marketplace_status()
    }


# =========================================================
# 17. IMAGE SEARCH API
# =========================================================

@app.post("/api/v1/search/image")
async def search_image(file: UploadFile = File(...)):

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    image_bytes = await file.read()

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    if len(image_bytes) > MAX_IMAGE_SIZE:

        raise HTTPException(
            status_code=413,
            detail="Image size must be less than 10 MB."
        )

    try:

        image = Image.open(io.BytesIO(image_bytes))
        image.verify()

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Image file is invalid or corrupted."
        )

    product = identify_product(
        image_bytes,
        file.content_type
    )

    return build_product_response(product)


# =========================================================
# 18. SMART CANDIDATE MATCHING API
# =========================================================

@app.post("/api/v1/match/candidate")
def match_candidate(data: CandidateMatchRequest):

    product = Product(
        brand=data.brand,
        model=data.model,
        variant=data.variant,
        confidence=1.0
    )

    allowed_names = [
        item["name"]
        for item in MARKETPLACE_CONFIG
    ]

    if data.marketplace not in allowed_names:

        raise HTTPException(
            status_code=400,
            detail="Unsupported marketplace."
        )

    if not validate_direct_url(
        data.marketplace,
        data.product_url
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid marketplace URL."
        )

    if not data.candidate_title.strip():

        raise HTTPException(
            status_code=400,
            detail="Candidate product title is required."
        )

    match = compare_product_title(
        product,
        data.candidate_title
    )

    asin = None

    if data.marketplace == "Amazon":
        asin = extract_amazon_asin(data.product_url)

    return {
        "success": True,
        "marketplace": data.marketplace,
        "candidate_title": data.candidate_title,
        "candidate_url": data.product_url,
        "asin": asin,
        "title_match": match,
        "exact_match": False,
        "match_status": (
            "possible_match_needs_verification"
            if match["possible_match"]
            else "not_confirmed"
        ),
        "message": (
            "Title looks similar, but product identity "
            "must be confirmed using a trusted product source."
        )
    }


# =========================================================
# 19. DIRECT LINK FORMAT CHECK API
# =========================================================

@app.post("/api/v1/direct-link/verify")
def register_direct_link(data: DirectLinkRequest):

    product = Product(
        brand=data.brand,
        model=data.model,
        variant=data.variant,
        confidence=1.0
    )

    allowed_names = [
        item["name"]
        for item in MARKETPLACE_CONFIG
    ]

    if data.marketplace not in allowed_names:

        raise HTTPException(
            status_code=400,
            detail="Unsupported marketplace."
        )

    if not validate_direct_url(
        data.marketplace,
        data.product_url
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid marketplace URL."
        )

    title_match = None

    if data.candidate_title.strip():

        title_match = compare_product_title(
            product,
            data.candidate_title
        )

    asin = None

    if data.marketplace == "Amazon":
        asin = extract_amazon_asin(data.product_url)

    return {
        "success": True,
        "message": (
            "URL domain and format accepted. "
            "Product identity is not independently verified."
        ),
        "marketplace": data.marketplace,
        "product_url": data.product_url,
        "asin": asin,
        "title_match": title_match,
        "exact_match": False,
        "verified": False
    }


# =========================================================
# 20. API INFORMATION
# =========================================================

@app.get("/api/v1/info")
def api_info():

    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "features": [
            "Gemini image identification",
            "Marketplace search links",
            "Product title similarity matching",
            "Amazon ASIN extraction",
            "Marketplace URL domain validation",
            "Exact offer response structure"
        ],
        "limitations": [
            "Live marketplace prices are not connected",
            "Automatic marketplace product discovery is not connected",
            "Title similarity is not proof of exact product identity",
            "Direct product identity verification requires trusted product data"
        ]
    }
    
