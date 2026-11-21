
import os
import io
import json
import base64
import re
import requests

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="PriceDekho AI",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

MAX_SIZE = 8 * 1024 * 1024

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}


def identify_product(image_bytes, mime_type):

    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key is not configured."
        )

    prompt = """
    Identify the consumer product in this image.

    Return only valid JSON:
    {
      "brand": "brand name or unknown",
      "model": "exact model or unknown",
      "category": "product category",
      "confidence": 0.0,
      "evidence": "visible identifying clues"
    }

    Rules:
    1. Do not guess an exact model.
    2. If the model is not identifiable, return unknown.
    3. Do not confuse similar-looking products.
    4. Confidence must be between 0 and 1.
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
            headers={"x-goog-api-key": API_KEY},
            json=payload,
            timeout=90
        )

        response.raise_for_status()

        data = response.json()

        result_text = (
            data["candidates"][0]
            ["content"]["parts"][0]["text"]
        )

        result_text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            result_text.strip()
        )

        product = json.loads(result_text)

        confidence = float(
            product.get("confidence", 0)
        )

        return {
            "brand": str(
                product.get("brand", "unknown")
            ).strip(),
            "model": str(
                product.get("model", "unknown")
            ).strip(),
            "category": str(
                product.get("category", "unknown")
            ).strip(),
            "confidence": max(
                0.0, min(1.0, confidence)
            ),
            "evidence": str(
                product.get("evidence", "")
            ).strip()
        }

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code
        error_text = e.response.text[:2000]

        print("GEMINI HTTP STATUS:", status)
        print("GEMINI ERROR RESPONSE:", error_text)

        raise HTTPException(
            status_code=502,
            detail={
                "message": "Gemini API rejected the request.",
                "gemini_status": status,
                "gemini_error": error_text
            }
        )

    except requests.exceptions.RequestException as e:
        print("GEMINI CONNECTION ERROR:", str(e))

        raise HTTPException(
            status_code=502,
            detail="Could not connect to Gemini. Check Render logs."
        )

    except (KeyError, IndexError, ValueError, TypeError) as e:
        print("GEMINI RESPONSE ERROR:", str(e))

        raise HTTPException(
            status_code=502,
            detail="Gemini returned an invalid response."
        )


def find_exact_offers(product):
    # Marketplace APIs will be connected later.
    # Never return fake prices.
    return []


@app.get("/")
def home():
    return {
        "app": "PriceDekho AI",
        "status": "running",
        "version": "1.1.0"
    }


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "ai_configured": bool(API_KEY),
        "model": MODEL,
        "service": "PriceDekho AI Backend"
    }


@app.post("/api/v1/search/image")
async def search_image(
    file: UploadFile = File(...)
):

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload JPG, PNG or WEBP only."
        )

    image_bytes = await file.read(MAX_SIZE + 1)

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

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        )
        image.verify()

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    product = identify_product(
        image_bytes,
        file.content_type
    )

    brand = product["brand"].lower()
    model = product["model"].lower()

    exact_match = (
        brand not in ("unknown", "")
        and model not in ("unknown", "")
        and product["confidence"] >= 0.75
    )

    if not exact_match:
        return {
            "success": True,
            "exact_match": False,
            "message": "Exact model could not be confirmed. Try a clearer photo.",
            "product": product,
            "offers": []
        }

    offers = find_exact_offers(product)

    return {
        "success": True,
        "exact_match": True,
        "message": (
            "Product identified. Marketplace feeds are not connected yet."
            if not offers
            else "Exact product offers found."
        ),
        "product": product,
        "offers": offers
    }
    
