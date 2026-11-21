import os
import io
import json
import base64
import re
import requests

from urllib.parse import quote_plus

from PIL import Image, UnidentifiedImageError

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware


# ==========================================
# PRICEDEKHO AI - VERSION 3.0
# ==========================================

app = FastAPI(
    title="PriceDekho AI",
    description="AI Product Identification and Marketplace Redirect API",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# CONFIGURATION
# ==========================================

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

MAX_SIZE = 8 * 1024 * 1024

MIN_CONFIDENCE = 0.75

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}


# ==========================================
# MARKETPLACE INFORMATION
# ==========================================

MARKETPLACES = {
    "amazon": "Amazon",
    "flipkart": "Flipkart",
    "meesho": "Meesho",
    "google_shopping": "Google Shopping",
    "myntra": "Myntra",
    "ajio": "AJIO",
    "tata_cliq": "Tata CLiQ"
}
# ==========================================
# TEXT HELPERS
# ==========================================

def clean_text(value, default="unknown"):

    if value is None:
        return default

    value = re.sub(
        r"\s+",
        " ",
        str(value)
    ).strip()

    return value if value else default


def normalize_text(value):

    value = clean_text(value, "")

    return re.sub(
        r"[^a-z0-9]+",
        "",
        value.lower()
    )


# ==========================================
# GEMINI PRODUCT IDENTIFICATION
# ==========================================

def identify_product(image_bytes, mime_type):

    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key is not configured."
        )

    prompt = """
    Identify the consumer product in this image.

    Return ONLY valid JSON:

    {
      "brand": "brand name or unknown",
      "model": "exact model or unknown",
      "category": "product category",
      "variant": "color, size or variant, otherwise unknown",
      "confidence": 0.0,
      "evidence": "visible identifying clues"
    }

    Rules:

    1. Identify the exact brand and model only
       when supported by visible evidence.

    2. Never guess an exact model.

    3. Do not confuse similar-looking products.

    4. If exact model is not identifiable,
       return unknown.

    5. Do not invent product specifications.

    6. Confidence must be between 0 and 1.

    7. Return valid JSON only.
    """

    endpoint = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{MODEL}:generateContent"
    )

    payload = {
        "contents": [{
            "parts": [
                {
                    "text": prompt
                },
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(
                            image_bytes
                        ).decode("utf-8")
                    }
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json"
        }
    }

    try:

        response = requests.post(
            endpoint,
            headers={
                "x-goog-api-key": API_KEY
            },
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        candidates = data.get(
            "candidates",
            []
        )

        if not candidates:
            raise ValueError(
                "Gemini returned no candidates."
            )

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )

        if not parts or "text" not in parts[0]:
            raise ValueError(
                "Gemini returned no text."
            )

        result_text = parts[0]["text"].strip()

        result_text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            result_text
        )

        product = json.loads(result_text)

        confidence = float(
            product.get("confidence", 0)
        )

        if not 0 <= confidence <= 1:
            confidence = 0

        return {
            "brand": clean_text(
                product.get("brand")
            ),
            "model": clean_text(
                product.get("model")
            ),
            "category": clean_text(
                product.get("category")
            ),
            "variant": clean_text(
                product.get("variant")
            ),
            "confidence": confidence,
            "evidence": clean_text(
                product.get("evidence"),
                ""
            )
        }

    except requests.exceptions.RequestException:

        raise HTTPException(
            status_code=502,
            detail="Could not connect to Gemini."
        )

    except (
        KeyError,
        IndexError,
        ValueError,
        TypeError,
        json.JSONDecodeError
    ):

        raise HTTPException(
            status_code=502,
            detail="Invalid AI response."
        )


# ==========================================
# EXACT PRODUCT VERIFICATION
# ==========================================

def is_exact_product(product):

    brand = normalize_text(
        product.get("brand")
    )

    model = normalize_text(
        product.get("model")
    )

    confidence = product.get(
        "confidence",
        0
    )

    if not brand or not model:
        return False

    if brand == "unknown" or model == "unknown":
        return False

    if confidence < MIN_CONFIDENCE:
        return False

    return True
    
# ==========================================
# MARKETPLACE DIRECT SEARCH LINKS
# ==========================================

def create_marketplace_links(product):

    brand = clean_text(
        product.get("brand"),
        ""
    )

    model = clean_text(
        product.get("model"),
        ""
    )

    variant = clean_text(
        product.get("variant"),
        ""
    )

    # Build product query.

    query = f"{brand} {model}"

    if variant and variant.lower() != "unknown":
        query += f" {variant}"

    query = re.sub(
        r"\s+",
        " ",
        query
    ).strip()

    q = quote_plus(query)

    return {
        "amazon": {
            "name": "Amazon",
            "url": f"https://www.amazon.in/s?k={q}",
            "link_type": "search"
        },

        "flipkart": {
            "name": "Flipkart",
            "url": f"https://www.flipkart.com/search?q={q}",
            "link_type": "search"
        },

        "meesho": {
            "name": "Meesho",
            "url": f"https://www.meesho.com/search?q={q}",
            "link_type": "search"
        },

        "google_shopping": {
            "name": "Google Shopping",
            "url": (
                "https://www.google.com/search"
                f"?tbm=shop&q={q}"
            ),
            "link_type": "search"
        },

        "myntra": {
            "name": "Myntra",
            "url": (
                "https://www.myntra.com/"
                f"search?query={q}"
            ),
            "link_type": "search"
        },

        "ajio": {
            "name": "AJIO",
            "url": (
                "https://www.ajio.com/search/"
                f"?text={q}"
            ),
            "link_type": "search"
        },

        "tata_cliq": {
            "name": "Tata CLiQ",
            "url": (
                "https://www.tatacliq.com/search/"
                f"?searchCategory=all&text={q}"
            ),
            "link_type": "search"
        }
    }


# ==========================================
# MARKETPLACE STATUS
# ==========================================

def get_marketplace_status():

    result = {}

    for key, name in MARKETPLACES.items():

        result[key] = {
            "name": name,
            "status": "search_link_available",
            "price_status": "not_connected"
        }

    return result


# ==========================================
# PRICE COMPARISON
# ==========================================

def find_exact_offers(product):

    """
    Marketplace price APIs are not connected yet.

    This function must return only genuine,
    verified marketplace offers.

    No fake prices are generated.
    """

    return []


# ==========================================
# CHEAPEST OFFER
# ==========================================

def find_cheapest_offer(offers):

    valid_offers = []

    for offer in offers:

        if offer.get("exact_match") is not True:
            continue

        price = offer.get("price")

        try:
            price = float(price)
        except (TypeError, ValueError):
            continue

        if price <= 0:
            continue

        valid_offers.append(offer)

    if not valid_offers:
        return None

    return min(
        valid_offers,
        key=lambda offer: float(offer["price"])
    )


# ==========================================
# RESPONSE BUILDER
# ==========================================

def build_product_response(product):

    offers = find_exact_offers(product)

    cheapest = find_cheapest_offer(
        offers
    )

    marketplace_links = create_marketplace_links(
        product
    )

    return {
        "success": True,
        "exact_match": True,
        "message": (
            "Product identified. Marketplace search links are ready."
            if not offers
            else "Verified product offers found."
        ),
        "product": product,
        "offers": offers,
        "cheapest_offer": cheapest,
        "marketplace_links": marketplace_links,
        "marketplaces": get_marketplace_status()
    }
    
# ==========================================
# HOME API
# ==========================================

@app.get("/")
def home():

    return {
        "app": "PriceDekho AI",
        "status": "running",
        "version": "3.0.0",
        "message": "PriceDekho AI backend is working."
    }


# ==========================================
# HEALTH API
# ==========================================

@app.get("/api/v1/health")
def health():

    return {
        "status": "ok",
        "ai_configured": bool(API_KEY),
        "service": "PriceDekho AI Backend",
        "version": "3.0.0"
    }


# ==========================================
# MARKETPLACE STATUS API
# ==========================================

@app.get("/api/v1/marketplaces")
def marketplaces():

    return {
        "success": True,
        "marketplaces": get_marketplace_status(),
        "total_platforms": len(MARKETPLACES),
        "price_comparison": "not_connected",
        "message": (
            "Marketplace search links are available. "
            "Live price APIs are not connected."
        )
    }


# ==========================================
# PRODUCT LINK API
# ==========================================

@app.get("/api/v1/product-links")
def product_links(
    brand: str,
    model: str,
    variant: str = "unknown"
):

    product = {
        "brand": clean_text(brand),
        "model": clean_text(model),
        "variant": clean_text(variant)
    }

    if (
        normalize_text(product["brand"]) == "unknown"
        or normalize_text(product["model"]) == "unknown"
    ):
        raise HTTPException(
            status_code=400,
            detail="Valid brand and model are required."
        )

    links = create_marketplace_links(
        product
    )

    return {
        "success": True,
        "product": product,
        "marketplace_links": links
}
# ==========================================
# IMAGE PRODUCT SEARCH API
# ==========================================

@app.post("/api/v1/search/image")
async def search_image(
    file: UploadFile = File(...)
):

    # Check image MIME type.

    if file.content_type not in ALLOWED_TYPES:

        raise HTTPException(
            status_code=415,
            detail="Upload JPG, PNG or WEBP only."
        )

    # Read image with size limit.

    image_bytes = await file.read(
        MAX_SIZE + 1
    )

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Image is empty."
        )

    if len(image_bytes) > MAX_SIZE:

        raise HTTPException(
            status_code=413,
            detail="Image must be under 8 MB."
        )

    # Verify actual image.

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.verify()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    # Identify product using Gemini.

    product = identify_product(
        image_bytes,
        file.content_type
    )

    # Verify exact product.

    exact_match = is_exact_product(
        product
    )

    # Stop if product identification is uncertain.

    if not exact_match:

        return {
            "success": True,
            "exact_match": False,
            "message": (
                "Exact model could not be confirmed. "
                "Please upload a clearer product photo."
            ),
            "product": product,
            "offers": [],
            "cheapest_offer": None,
            "marketplace_links": {},
            "marketplaces": get_marketplace_status()
        }

    # Generate complete response.

    return build_product_response(
        product
    )
    
