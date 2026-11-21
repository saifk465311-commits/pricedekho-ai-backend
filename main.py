
import os
import io
import json
import re
import logging
from urllib.parse import quote_plus
from typing import Optional, List, Dict, Any

import requests
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pricedekho")

APP_NAME = "PriceDekho AI"
APP_VERSION = "4.0.0"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

MAX_IMAGE_SIZE = 10 * 1024 * 1024

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="AI Product Identification and Marketplace Link Matching API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class Marketplace(BaseModel):
    name: str
    url: str
    link_type: str = "search"
    exact_match: bool = False
    
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
- If model is unclear, return empty model.
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
                            "data": __import__("base64").b64encode(
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

        text = result["candidates"][0]["content"]["parts"][0]["text"]
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

    except Exception as error:
        logger.exception("Gemini identification failed")

        raise HTTPException(
            status_code=502,
            detail="Product identification failed: " + str(error)
        )


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
MARKETPLACE_CONFIG = [
    {
        "name": "Amazon",
        "search": "https://www.amazon.in/s?k={query}",
        "direct": "https://www.amazon.in/dp/{product_id}"
    },
    {
        "name": "Flipkart",
        "search": "https://www.flipkart.com/search?q={query}",
        "direct": None
    },
    {
        "name": "Meesho",
        "search": "https://www.meesho.com/search?q={query}",
        "direct": None
    },
    {
        "name": "Google Shopping",
        "search": "https://www.google.com/search?tbm=shop&q={query}",
        "direct": None
    },
    {
        "name": "Myntra",
        "search": "https://www.myntra.com/search?query={query}",
        "direct": None
    },
    {
        "name": "AJIO",
        "search": "https://www.ajio.com/search/?text={query}",
        "direct": None
    },
    {
        "name": "Tata CLiQ",
        "search": "https://www.tatacliq.com/search/?searchCategory=all&text={query}",
        "direct": None
    }
]


def validate_direct_url(name: str, url: str) -> bool:

    allowed_domains = {
        "Amazon": ["amazon.in"],
        "Flipkart": ["flipkart.com"],
        "Meesho": ["meesho.com"],
        "Google Shopping": ["google.com"],
        "Myntra": ["myntra.com"],
        "AJIO": ["ajio.com"],
        "Tata CLiQ": ["tatacliq.com"]
    }

    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)

        if parsed.scheme != "https":
            return False

        host = (parsed.hostname or "").lower()

        return any(
            host == domain or host.endswith("." + domain)
            for domain in allowed_domains.get(name, [])
        )

    except Exception:
        return False


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

        if (
            direct_url
            and validate_direct_url(name, direct_url)
        ):
            links.append({
                "name": name,
                "url": direct_url,
                "link_type": "direct",
                "exact_match": True
            })

        else:

            search_url = marketplace["search"].format(
                query=query
            )

            links.append({
                "name": name,
                "url": search_url,
                "link_type": "search",
                "exact_match": False
            })

    return links


def create_amazon_direct_url(asin: str) -> Optional[str]:

    asin = str(asin).strip().upper()

    if not re.fullmatch(r"[A-Z0-9]{10}", asin):
        return None

    return "https://www.amazon.in/dp/" + asin


def get_marketplace_status():

    result = []

    for item in MARKETPLACE_CONFIG:

        result.append({
            "name": item["name"],
            "status": "search_link_available",
            "direct_link_status": "verified_data_required",
            "price_status": "not_connected"
        })

    return result
def find_exact_offers(product: Product) -> List[Dict[str, Any]]:

    # No marketplace price feeds are connected yet.
    # Never invent product prices or direct product URLs.

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
        "product": product.model_dump(),
        "offers": offers,
        "cheapest_offer": cheapest_offer,
        "marketplace_links": marketplace_links,
        "marketplaces": get_marketplace_status()
    }


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


@app.get("/api/v1/marketplaces")
def marketplaces():

    return {
        "success": True,
        "marketplaces": get_marketplace_status()
    }


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
        "product": product.model_dump(),
        "marketplace_links": create_marketplace_links(product),
        "marketplaces": get_marketplace_status()
    }


@app.post("/api/v1/search/image")
async def search_image(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
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
class DirectLinkRequest(BaseModel):
    brand: str
    model: str
    variant: str = ""
    marketplace: str
    product_url: str


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
            detail="Invalid marketplace product URL."
        )

    # This endpoint validates the URL domain and format.
    # It does NOT independently verify product identity.
    # Only use it after a trusted source confirms the exact match.

    links = create_marketplace_links(
        product,
        {
            data.marketplace: data.product_url
        }
    )

    return {
        "success": True,
        "message": "URL format accepted. Product identity still requires verification.",
        "marketplace_links": links
    }
    
