
import os
import io
import json
import base64
import re
import requests

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

app = FastAPI(
    title="PriceDekho AI",
    description="AI Product Identification and Price Comparison API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

MAX_SIZE = 8 * 1024 * 1024

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}

MIN_CONFIDENCE = 0.75


# --------------------------------------------------
# MARKETPLACE CONNECTOR STATUS
# --------------------------------------------------

MARKETPLACES = {
    "amazon": {
        "name": "Amazon",
        "status": "not_connected"
    },
    "flipkart": {
        "name": "Flipkart",
        "status": "not_connected"
    },
    "meesho": {
        "name": "Meesho",
        "status": "not_connected"
    },
    "google_shopping": {
        "name": "Google Shopping",
        "status": "not_connected"
    },
    "myntra": {
        "name": "Myntra",
        "status": "not_connected"
    },
    "ajio": {
        "name": "AJIO",
        "status": "not_connected"
    },
    "tata_cliq": {
        "name": "Tata CLiQ",
        "status": "not_connected"
    }
}


# --------------------------------------------------
# TEXT HELPERS
# --------------------------------------------------

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


# --------------------------------------------------
# GEMINI PRODUCT IDENTIFICATION
# --------------------------------------------------

def identify_product(image_bytes, mime_type):

    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key is not configured."
        )

    prompt = """
    Identify the consumer product in this image.

    Return ONLY valid JSON with these fields:

    {
      "brand": "brand name or unknown",
      "model": "exact model or unknown",
      "category": "product category",
      "variant": "exact variant, color, size or unknown",
      "confidence": 0.0,
      "evidence": "visible identifying clues"
    }

    Important rules:

    1. Identify the exact brand and model only when supported
       by visible evidence.

    2. Never guess a model from appearance alone.

    3. Do not confuse similar-looking products.

    4. If the exact model cannot be confirmed, return unknown.

    5. Do not invent product specifications or variants.

    6. Confidence must be between 0 and 1.

    7. Return JSON only. No markdown.
    """

    endpoint = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{MODEL}:generateContent"
    )

    payload = {
        "contents": [{
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

        candidates = data.get("candidates", [])

        if not candidates:
            raise ValueError("Gemini returned no candidates.")

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )

        if not parts or "text" not in parts[0]:
            raise ValueError("Gemini returned no text.")

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


# --------------------------------------------------
# EXACT PRODUCT MATCHING
# --------------------------------------------------

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


# --------------------------------------------------
# MARKETPLACE PRICE CONNECTORS
# --------------------------------------------------

def get_marketplace_status():

    return {
        key: {
            "name": value["name"],
            "status": value["status"]
        }
        for key, value in MARKETPLACES.items()
    }


def find_exact_offers(product):

    """
    Future integration point for authorized marketplace APIs.

    Only verified product listings should be returned.

    Expected offer format:

    {
        "marketplace": "Amazon",
        "brand": "boAt",
        "model": "Airdopes 141 Pro",
        "price": 999,
        "currency": "INR",
        "original_price": 1499,
        "availability": "in_stock",
        "product_url": "https://example.com/product",
        "exact_match": true
    }

    No fake prices or invented product URLs.
    """

    # No marketplace API is connected yet.
    return []


# --------------------------------------------------
# HEALTH AND INFORMATION
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "app": "PriceDekho AI",
        "status": "running",
        "version": "2.0.0"
    }


@app.get("/api/v1/health")
def health():

    return {
        "status": "ok",
        "ai_configured": bool(API_KEY),
        "service": "PriceDekho AI Backend",
        "version": "2.0.0"
    }


@app.get("/api/v1/marketplaces")
def marketplaces():

    return {
        "success": True,
        "marketplaces": get_marketplace_status(),
        "connected_count": 0,
        "message": (
            "Marketplace APIs are not connected yet."
        )
    }


# --------------------------------------------------
# IMAGE PRODUCT SEARCH
# --------------------------------------------------

@app.post("/api/v1/search/image")
async def search_image(
    file: UploadFile = File(...)
):

    # Validate file type.

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

    # Validate actual image content.

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

    exact_match = is_exact_product(
        product
    )

    # If product is not confidently identified.

    if not exact_match:

        return {
            "success": True,
            "exact_match": False,
            "message": (
                "Exact model could not be confirmed. "
                "Try a clearer photo."
            ),
            "product": product,
            "offers": [],
            "marketplaces": get_marketplace_status()
        }

    # Find real marketplace offers.

    offers = find_exact_offers(
        product
    )

    # Final response.

    return {
        "success": True,
        "exact_match": True,
        "message": (
            "Product identified. Marketplace APIs "
            "are not connected yet."
            if not offers
            else "Exact product offers found."
        ),
        "product": product,
        "offers": offers,
        "marketplaces": get_marketplace_status()
    }
