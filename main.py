import os
import base64
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

load_dotenv()

# =========================================================
# BEHRAD IMAGE AI
# Cloudflare Workers AI backend
# =========================================================

APP_NAME = "BEHRAD IMAGE AI"
APP_VERSION = "2.0.0"

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()

CLOUDFLARE_MODEL = "@cf/black-forest-labs/flux-1-schnell"

CLOUDFLARE_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/"
    f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/"
    f"{CLOUDFLARE_MODEL}"
)

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
)

# =========================================================
# Simple in-memory rate limiter
# =========================================================

RATE_LIMIT_REQUESTS = 8
RATE_LIMIT_WINDOW = 60

request_history = defaultdict(deque)


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")

    if forwarded:
        return forwarded.split(",")[0].strip()

    if request.client:
        return request.client.host

    return "unknown"


def rate_limit_ok(ip: str) -> bool:
    now = time.time()
    history = request_history[ip]

    while history and now - history[0] > RATE_LIMIT_WINDOW:
        history.popleft()

    if len(history) >= RATE_LIMIT_REQUESTS:
        return False

    history.append(now)
    return True


# =========================================================
# Request model
# =========================================================

class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2048,
    )

    width: int = Field(
        default=768,
        ge=512,
        le=1024,
    )

    height: int = Field(
        default=768,
        ge=512,
        le=1024,
    )

    seed: int = Field(
        default=-1,
        ge=-1,
        le=2147483647,
    )

    steps: int = Field(
        default=4,
        ge=1,
        le=8,
    )

    enhance: bool = False
    safe: bool = True


# =========================================================
# Security headers
# =========================================================

@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=()"
    )

    return response


# =========================================================
# Request body limit
# =========================================================

@app.middleware("http")
async def request_size_limit(request: Request, call_next):
    content_length = request.headers.get("content-length")

    if content_length:
        try:
            if int(content_length) > 32768:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "Request body is too large."
                    },
                )
        except ValueError:
            pass

    return await call_next(request)


# =========================================================
# Frontend
# =========================================================

@app.get("/")
async def home():
    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():
        return JSONResponse(
            status_code=500,
            content={
                "error": "static/index.html not found."
            },
        )

    return FileResponse(index_file)


# =========================================================
# Health
# =========================================================

@app.get("/api/health")
async def health():
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "ok",
        "provider": "Cloudflare Workers AI",
        "model": CLOUDFLARE_MODEL,
        "cloudflare_configured": bool(
            CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN
        ),
    }


# =========================================================
# Models
# =========================================================

@app.get("/api/models")
async def models():
    return {
        "models": [
            {
                "id": CLOUDFLARE_MODEL,
                "name": "FLUX.1 Schnell",
                "provider": "Cloudflare Workers AI",
                "type": "text-to-image",
            }
        ]
    }


# =========================================================
# Generate image
# =========================================================

@app.post("/api/generate")
async def generate_image(
    payload: GenerateRequest,
    request: Request,
):
    # -----------------------------------------------------
    # Configuration check
    # -----------------------------------------------------

    if not CLOUDFLARE_ACCOUNT_ID:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Cloudflare Account ID is not configured."
            },
        )

    if not CLOUDFLARE_API_TOKEN:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Cloudflare API Token is not configured."
            },
        )

    # -----------------------------------------------------
    # Rate limit
    # -----------------------------------------------------

    ip = get_client_ip(request)

    if not rate_limit_ok(ip):
        return JSONResponse(
            status_code=429,
            content={
                "error": "Too many requests. Please wait a minute."
            },
        )

    # -----------------------------------------------------
    # Prompt cleanup
    # -----------------------------------------------------

    prompt = payload.prompt.strip()

    if not prompt:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Prompt cannot be empty."
            },
        )

    # -----------------------------------------------------
    # Seed
    # -----------------------------------------------------

    seed: Optional[int]

    if payload.seed == -1:
        seed = None
    else:
        seed = payload.seed

    # -----------------------------------------------------
    # Cloudflare request
    # -----------------------------------------------------

    body = {
        "prompt": prompt,
        "steps": payload.steps,
    }

    if seed is not None:
        body["seed"] = seed

    # -----------------------------------------------------
    # Call Cloudflare
    # -----------------------------------------------------

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=20.0,
                read=180.0,
                write=30.0,
                pool=20.0,
            )
        ) as client:

            response = await client.post(
                CLOUDFLARE_URL,
                headers={
                    "Authorization": (
                        f"Bearer {CLOUDFLARE_API_TOKEN}"
                    ),
                    "Content-Type": "application/json",
                },
                json=body,
            )

    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={
                "error": (
                    "Cloudflare image generation timed out. "
                    "Please try again."
                )
            },
        )

    except httpx.RequestError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "error": (
                    "Could not connect to Cloudflare Workers AI."
                ),
                "details": str(exc),
            },
        )

    # -----------------------------------------------------
    # Cloudflare HTTP errors
    # -----------------------------------------------------

    if response.status_code != 200:
        error_text = response.text[:2000]

        if response.status_code == 401:
            message = (
                "Cloudflare API Token is invalid or unauthorized."
            )

        elif response.status_code == 403:
            message = (
                "Cloudflare rejected the request. "
                "Check Workers AI permissions."
            )

        elif response.status_code == 429:
            message = (
                "Cloudflare rate limit or free quota limit "
                "has been reached. Please try again later."
            )

        elif response.status_code == 400:
            message = (
                "Cloudflare rejected the image request. "
                "Check the prompt or model parameters."
            )

        else:
            message = (
                f"Cloudflare returned HTTP {response.status_code}."
            )

        return JSONResponse(
            status_code=502,
            content={
                "error": message,
                "provider_status": response.status_code,
                "provider_response": error_text,
            },
        )

    # -----------------------------------------------------
    # Parse response
    # -----------------------------------------------------

    try:
        data = response.json()
    except Exception:
        return JSONResponse(
            status_code=502,
            content={
                "error": "Cloudflare returned an invalid response."
            },
        )

    if not data.get("success", True):
        return JSONResponse(
            status_code=502,
            content={
                "error": "Cloudflare image generation failed.",
                "provider_response": data,
            },
        )

    result = data.get("result")

    if not isinstance(result, dict):
        return JSONResponse(
            status_code=502,
            content={
                "error": "Cloudflare returned no image result."
            },
        )

    image_base64 = result.get("image")

    if not image_base64:
        return JSONResponse(
            status_code=502,
            content={
                "error": "Cloudflare returned an empty image."
            },
        )

    # -----------------------------------------------------
    # Validate Base64
    # -----------------------------------------------------

    try:
        image_bytes = base64.b64decode(
            image_base64,
            validate=True,
        )
    except Exception:
        return JSONResponse(
            status_code=502,
            content={
                "error": "Cloudflare returned invalid image data."
            },
        )

    # Prevent unexpectedly huge responses
    if len(image_bytes) > 15 * 1024 * 1024:
        return JSONResponse(
            status_code=502,
            content={
                "error": "Generated image is unexpectedly large."
            },
        )

    # -----------------------------------------------------
    # Return data URI
    # -----------------------------------------------------

    image_data_url = (
        "data:image/jpeg;base64,"
        + base64.b64encode(image_bytes).decode("ascii")
    )

    return {
        "success": True,
        "image": image_data_url,
        "model": CLOUDFLARE_MODEL,
        "prompt": prompt,
        "seed": seed,
        "steps": payload.steps,
        "width": payload.width,
        "height": payload.height,
    }


# =========================================================
# Static files
# =========================================================

if STATIC_DIR.exists():

    from fastapi.staticfiles import StaticFiles

    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static",
    )
