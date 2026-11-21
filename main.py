
import os
import io
import re
import json
import base64
import logging
from urllib.parse import quote_plus, urlparse
from typing import Optional, List

import requests
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# =========================================================
# CONFIGURATION
# =========================================================

APP_NAME = "PriceDekho AI Backend"
APP_VERSION = "4.2.0"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

GEMINI_TIMEOUT = 60
MAX_IMAGE_SIZE = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pricedekho")


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title=APP_NAME,
    description="AI-powered product identification and marketplace comparison backend.",
    version=APP_VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# PRODUCT MODEL
# =========================================================

class Product(BaseModel):
    brand: str = "Unknown"
    model: str = "Unknown"
    category: str = "General"
    variant: str = ""
    confidence: float = 0.0
    evidence: List[str] = Field(default_factory=list)


class DirectLinkRequest(BaseModel):
    brand: str
    model: str
    variant: Optional[str] = ""
    url: str
    title: Optional[str] = ""


class CandidateRequest(BaseModel):
    brand: str
    model: str
    variant: Optional[str] = ""
    title: str
    url: Optional[str] = ""


# =========================================================
# MARKETPLACE CONFIGURATION
# =========================================================

MARKETPLACES = [
    {
        "name": "Amazon",
        "domain": "amazon.in",
        "search_url": "https://www.amazon.in/s?k={query}"
    },
    {
        "name": "Flipkart",
        "domain": "flipkart.com",
        "search_url": "https://www.flipkart.com/search?q={query}"
    },
    {
        "name": "Meesho",
        "domain": "meesho.com",
        "search_url": "https://www.meesho.com/search?q={query}"
    },
    {
        "name": "Google Shopping",
        "domain": "google.com",
        "search_url": "https://www.google.com/search?tbm=shop&q={query}"
    },
    {
        "name": "Myntra",
        "domain": "myntra.com",
        "search_url": "https://www.myntra.com/{query}"
    },
    {
        "name": "AJIO",
        "domain": "ajio.com",
        "search_url": "https://www.ajio.com/search/?text={query}"
    },
    {
        "name": "Tata CLiQ",
        "domain": "tatacliq.com",
        "search_url": "https://www.tatacliq.com/search/?searchCategory=all&text={query}"
    }
]


# =========================================================
# GENERAL HELPERS
# =========================================================

def model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def normalize_text(value: str) -> str:
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def product_search_query(
    brand: str,
    model: str,
    variant: str = ""
) -> str:
    parts = [brand, model, variant]
    return " ".join(
        str(part).strip()
        for part in parts
        if part and str(part).strip()
    )


def get_product_tokens(value: str) -> set:
    ignored = {
        "the", "and", "with", "for", "new",
        "in", "on", "of", "by", "a", "an"
    }
    return {
        token for token in normalize_text(value).split()
        if token not in ignored
    }


def compare_product_title(
    brand: str,
    model: str,
    variant: str,
    title: str
) -> dict:
    expected = get_product_tokens(
        product_search_query(brand, model, variant)
    )
    actual = get_product_tokens(title)

    if not expected or not actual:
        return {
            "similarity": 0.0,
            "brand_match": False,
            "model_match": False,
            "exact_match": False
        }

    common = expected.intersection(actual)
    similarity = len(common) / len(expected)

    brand_tokens = get_product_tokens(brand)
    model_tokens = get_product_tokens(model)

    brand_match = (
        not brand_tokens or brand_tokens.issubset(actual)
    )

    model_match = (
        bool(model_tokens) and model_tokens.issubset(actual)
    )

    exact_match = (
        brand_match
        and model_match
        and similarity >= 0.85
    )

    return {
        "similarity": round(similarity, 2),
        "brand_match": brand_match,
        "model_match": model_match,
        "exact_match": exact_match
    }


# =========================================================
# IMAGE VALIDATION
# =========================================================

def validate_image(image_bytes: bytes) -> str:
    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="No image uploaded."
        )

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Maximum size is 10 MB."
        )

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()

        image = Image.open(io.BytesIO(image_bytes))
        image_format = (image.format or "").upper()

        mime_types = {
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp"
        }

        if image_format not in mime_types:
            raise HTTPException(
                status_code=400,
                detail="Only JPG, PNG and WEBP images are supported."
            )

        return mime_types[image_format]

    except HTTPException:
        raise

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported image file."
        )


# =========================================================
# GEMINI PRODUCT IDENTIFICATION
# =========================================================

def identify_product(
    image_bytes: bytes,
    mime_type: str
) -> Product:

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY environment variable is missing.")
        raise HTTPException(
            status_code=500,
            detail="Gemini API key is not configured on the server."
        )

    if not GEMINI_MODEL:
        raise HTTPException(
            status_code=500,
            detail="Gemini model is not configured."
        )

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = """
You are an expert product identification AI.

Analyze the uploaded product image carefully.

Identify the product brand, exact model, category and variant.

Rules:
1. Do not invent a brand or model.
2. If the exact model is not visible, use "Unknown".
3. Read visible labels and model numbers carefully.
4. Do not guess specifications that are not visible.
5. Return only valid JSON.
6. Confidence must be a number between 0 and 1.

Required JSON format:

{
  "brand": "Brand name",
  "model": "Exact model name or Unknown",
  "category": "Product category",
  "variant": "Color, size, wattage or other visible variant",
  "confidence": 0.0,
  "evidence": ["Visible text or visual evidence"]
}
"""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_base64
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

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=GEMINI_TIMEOUT
        )

        # IMPORTANT: Log actual Gemini error response.
        if not response.ok:
            logger.error(
                "Gemini API Error | Status: %s | Response: %s",
                response.status_code,
                response.text[:1500]
            )

            raise HTTPException(
                status_code=502,
                detail=f"Gemini API Error: {response.status_code}"
            )

        result = response.json()

        candidates = result.get("candidates", [])

        if not candidates:
            logger.error(
                "Gemini returned no candidates: %s",
                json.dumps(result)[:1500]
            )
            raise HTTPException(
                status_code=502,
                detail="Gemini could not identify this image."
            )

        parts = candidates[0].get("content", {}).get("parts", [])

        if not parts or not parts[0].get("text"):
            logger.error(
                "Gemini returned empty text: %s",
                json.dumps(result)[:1500]
            )
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response."
            )

        raw_text = parts[0]["text"].strip()

        # Remove JSON markdown fences if present.
        raw_text = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw_text,
            flags=re.IGNORECASE
        )
        raw_text = re.sub(r"\s*```$", "", raw_text)

        product_data = json.loads(raw_text)

        product = Product(
            brand=str(product_data.get("brand", "Unknown")),
            model=str(product_data.get("model", "Unknown")),
            category=str(product_data.get("category", "General")),
            variant=str(product_data.get("variant", "")),
            confidence=max(
                0.0,
                min(1.0, float(product_data.get("confidence", 0.0)))
            ),
            evidence=product_data.get("evidence", [])
        )

        logger.info(
            "Product identified: %s %s",
            product.brand,
            product.model
        )

        return product

    except HTTPException:
        raise

    except requests.Timeout:
        logger.exception("Gemini request timed out.")
        raise HTTPException(
            status_code=504,
            detail="Gemini request timed out. Please try again."
        )

    except requests.RequestException:
        logger.exception("Gemini network request failed.")
        raise HTTPException(
            status_code=502,
            detail="Could not connect to Gemini API."
        )

    except (ValueError, KeyError, TypeError, IndexError):
        logger.exception("Gemini response parsing failed.")
        raise HTTPException(
            status_code=502,
            detail="Gemini returned an invalid product response."
        )

    except Exception:
        logger.exception("Unexpected Gemini identification failure.")
        raise HTTPException(
            status_code=502,
            detail="Product identification failed. Please try again."
        )


# =========================================================
# DIRECT URL VALIDATION
# =========================================================

def validate_direct_url(url: str) -> bool:
    if not url:
        return False

    try:
        parsed = urlparse(url.strip())

        if parsed.scheme != "https":
            return False

        hostname = (parsed.hostname or "").lower()

        allowed_domains = [
            "amazon.in",
            "flipkart.com",
            "meesho.com",
            "google.com",
            "myntra.com",
            "ajio.com",
            "tatacliq.com"
        ]

        return any(
            hostname == domain or hostname.endswith("." + domain)
            for domain in allowed_domains
        )

    except Exception:
        return False


def extract_amazon_asin(url: str) -> Optional[str]:
    if not validate_direct_url(url):
        return None

    parsed = urlparse(url)

    if not (
        parsed.hostname == "amazon.in"
        or parsed.hostname.endswith(".amazon.in")
    ):
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
            flags=re.IGNORECASE
        )

        if match:
            return match.group(1).upper()

    return None


def create_amazon_direct_url(asin: str) -> Optional[str]:
    if not asin:
        return None

    asin = asin.strip().upper()

    if not re.fullmatch(r"[A-Z0-9]{10}", asin):
        return None

    return f"https://www.amazon.in/dp/{asin}"


# =========================================================
# MARKETPLACE LINKS
# =========================================================

def create_marketplace_links(
    product: Product,
    verified_direct_urls: Optional[dict] = None
) -> list:

    verified_direct_urls = verified_direct_urls or {}

    query = quote_plus(
        product_search_query(
            product.brand,
            product.model,
            product.variant
        )
    )

    links = []

    for marketplace in MARKETPLACES:

        name = marketplace["name"]
        direct_url = verified_direct_urls.get(name)

        if direct_url and validate_direct_url(direct_url):
            links.append({
                "marketplace": name,
                "url": direct_url,
                "link_type": "direct_candidate",
                "exact_match": False,
                "match_status": "identity_not_verified",
                "price": None
            })

        else:
            search_url = marketplace["search_url"].format(
                query=query
            )

            links.append({
                "marketplace": name,
                "url": search_url,
                "link_type": "search",
                "exact_match": False,
                "match_status": "search_only",
                "price": None
            })

    return links


# =========================================================
# PRICE / OFFER PLACEHOLDER
# =========================================================

def find_exact_offers(product: Product) -> list:
    """
    Live marketplace prices are not connected yet.

    This function intentionally returns an empty list.
    Do not show invented prices or unverified offers.
    """
    return []


def get_cheapest_offer(offers: list):
    if not offers:
        return None

    valid_offers = [
        offer for offer in offers
        if offer.get("price") is not None
    ]

    if not valid_offers:
        return None

    return min(
        valid_offers,
        key=lambda offer: float(offer["price"])
    )


# =========================================================
# ROUTES
# =========================================================

@app.get("/")
def home():
    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
        "message": "PriceDekho AI backend is live."
    }


@app.get("/api/v1/health")
def health():
    return {
        "status": "healthy",
        "version": APP_VERSION,
        "gemini_configured": bool(GEMINI_API_KEY),
        "gemini_model": GEMINI_MODEL
    }


@app.get("/api/v1/info")
def info():
    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "gemini_model": GEMINI_MODEL,
        "gemini_configured": bool(GEMINI_API_KEY),
        "marketplaces": [
            item["name"] for item in MARKETPLACES
        ],
        "features": {
            "image_identification": True,
            "marketplace_search_links": True,
            "direct_url_validation": True,
            "live_prices": False,
            "verified_exact_product_discovery": False
        }
    }


@app.get("/api/v1/marketplaces")
def get_marketplaces():
    return {
        "count": len(MARKETPLACES),
        "marketplaces": [
            item["name"] for item in MARKETPLACES
        ]
    }


@app.get("/api/v1/product-links")
def get_product_links(
    brand: str,
    model: str,
    variant: str = ""
):
    if not brand.strip() or not model.strip():
        raise HTTPException(
            status_code=400,
            detail="Brand and model are required."
        )

    product = Product(
        brand=brand.strip(),
        model=model.strip(),
        variant=variant.strip(),
        category="General",
        confidence=1.0
    )

    links = create_marketplace_links(product)

    return {
        "success": True,
        "product": model_to_dict(product),
        "marketplace_links": links,
        "offers": [],
        "cheapest_offer": None,
        "price_status": "not_connected"
    }


@app.post("/api/v1/search/image")
async def search_by_image(
    file: UploadFile = File(...)
):
    image_bytes = await file.read()

    mime_type = validate_image(image_bytes)

    product = identify_product(
        image_bytes,
        mime_type
    )

    marketplace_links = create_marketplace_links(product)

    offers = find_exact_offers(product)

    return {
        "success": True,
        "app_version": APP_VERSION,
        "product": model_to_dict(product),
        "marketplace_links": marketplace_links,
        "offers": offers,
        "cheapest_offer": get_cheapest_offer(offers),
        "price_status": "not_connected",
        "matching_status": "product_identified_search_links_available",
        "message": (
            "Product identified. Marketplace search links are available. "
            "Live prices and verified exact product links are not connected yet."
        )
    }


# =========================================================
# PRODUCT TITLE MATCHING
# =========================================================

@app.post("/api/v1/match/candidate")
def match_candidate(request: CandidateRequest):

    result = compare_product_title(
        request.brand,
        request.model,
        request.variant or "",
        request.title
    )

    return {
        "success": True,
        "match": result,
        "exact_match": False,
        "verified": False,
        "message": (
            "Title similarity is only a candidate check. "
            "It does not prove that this is the exact product."
        )
    }


# =========================================================
# DIRECT LINK VERIFICATION
# =========================================================

@app.post("/api/v1/direct-link/verify")
def verify_direct_link(request: DirectLinkRequest):

    valid_domain = validate_direct_url(request.url)

    if not valid_domain:
        return {
            "success": False,
            "valid_domain": False,
            "exact_match": False,
            "verified": False,
            "message": "URL is invalid or marketplace domain is unsupported."
        }

    title_match = None

    if request.title:
        title_match = compare_product_title(
            request.brand,
            request.model,
            request.variant or "",
            request.title
        )

    asin = extract_amazon_asin(request.url)

    return {
        "success": True,
        "valid_domain": True,
        "exact_match": False,
        "verified": False,
        "asin": asin,
