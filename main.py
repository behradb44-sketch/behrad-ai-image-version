from __future__ import annotations

import base64
import os
import re
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator


# ============================================================
# Configuration
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

POLLINATIONS_BASE = "https://gen.pollinations.ai"
POLLINATIONS_IMAGE_URL = f"{POLLINATIONS_BASE}/image"

APP_NAME = "BEHRAD IMAGE AI"
APP_VERSION = "1.0.0"

MAX_PROMPT_LENGTH = 4000
MAX_REQUEST_BODY = 32 * 1024

GENERATION_TIMEOUT = httpx.Timeout(
    connect=15.0,
    read=180.0,
    write=30.0,
    pool=15.0,
)

# Simple in-memory protection.
# This is intentionally conservative for a public first release.
RATE_LIMIT_REQUESTS = 8
RATE_LIMIT_WINDOW = 60


# ============================================================
# App
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    docs_url="/api/docs",
    redoc_url=None,
)


# ============================================================
# Static files
# ============================================================

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static",
    )


# ============================================================
# Rate limiter
# ============================================================

_request_log: dict[str, deque[float]] = defaultdict(deque)


def get_client_ip(request: Request) -> str:
    """
    Get the apparent client IP.

    We intentionally do not blindly trust arbitrary forwarding headers.
    Render/proxy environments may provide X-Forwarded-For, but the
    application itself only uses the first value as a best-effort
    identifier for rate limiting.
    """
    forwarded = request.headers.get("x-forwarded-for")

    if forwarded:
        first = forwarded.split(",")[0].strip()

        if first:
            return first[:64]

    client = request.client

    if client and client.host:
        return client.host[:64]

    return "unknown"


def check_rate_limit(request: Request) -> None:
    now = time.monotonic()
    ip = get_client_ip(request)

    bucket = _request_log[ip]

    cutoff = now - RATE_LIMIT_WINDOW

    while bucket and bucket[0] <= cutoff:
        bucket.popleft()

    if len(bucket) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=(
                "درخواست‌های زیادی ارسال شده. "
                "چند لحظه صبر کن و دوباره امتحان کن."
            ),
            headers={"Retry-After": str(RATE_LIMIT_WINDOW)},
        )

    bucket.append(now)


# ============================================================
# Request models
# ============================================================

class GenerateRequest(BaseModel):
    prompt: str = Field(
        min_length=2,
        max_length=MAX_PROMPT_LENGTH,
    )

    model: str = Field(
        default="black-forest-labs/flux.2-pro",
        min_length=1,
        max_length=160,
    )

    width: int = Field(
        default=1024,
        ge=512,
        le=1536,
    )

    height: int = Field(
        default=1024,
        ge=512,
        le=1536,
    )

    seed: int = Field(
        default=-1,
        ge=-1,
        le=2147483647,
    )

    enhance: bool = False
    safe: bool = True

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        value = value.strip()

        # Remove control characters while preserving normal Unicode.
        value = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", value)

        # Collapse excessive whitespace.
        value = re.sub(r"\s+", " ", value).strip()

        if len(value) < 2:
            raise ValueError("Prompt is too short.")

        return value

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        value = value.strip()

        # Model IDs should not contain URL/query injection characters.
        if not re.fullmatch(r"[A-Za-z0-9._/@:-]+", value):
            raise ValueError("Invalid model identifier.")

        return value


# ============================================================
# Helpers
# ============================================================

def get_api_key() -> str:
    api_key = os.getenv("POLLINATIONS_API_KEY", "").strip()

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="POLLINATIONS_API_KEY is not configured on the server.",
        )

    return api_key


def safe_error_message(status_code: int) -> str:
    if status_code == 401:
        return "احراز هویت Pollinations ناموفق بود. API Key را بررسی کن."

    if status_code == 403:
        return "دسترسی Pollinations برای این درخواست مجاز نیست."

    if status_code == 404:
        return "مدل یا endpoint موردنظر پیدا نشد."

    if status_code == 408:
        return "تولید تصویر بیش از حد طول کشید."

    if status_code == 429:
        return "محدودیت درخواست Pollinations فعال شده. کمی بعد دوباره امتحان کن."

    if 500 <= status_code <= 599:
        return "سرور Pollinations موقتاً مشکل دارد. دوباره امتحان کن."

    return f"Pollinations request failed ({status_code})."


def validate_dimensions(width: int, height: int) -> None:
    pixels = width * height

    # Prevent accidentally huge generation requests.
    if pixels > 1536 * 1536:
        raise HTTPException(
            status_code=400,
            detail="ابعاد تصویر بیش از حد مجاز است.",
        )


# ============================================================
# Middleware
# ============================================================

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # Basic request-size protection.
    content_length = request.headers.get("content-length")

    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BODY:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body is too large."},
                )
        except ValueError:
            pass

    response = await call_next(request)

    # Security headers.
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=()"
    )

    return response


# ============================================================
# Routes
# ============================================================

@app.get("/", include_in_schema=False)
async def index():
    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=500,
            detail="Frontend files are missing.",
        )

    return FileResponse(index_file)


@app.get("/api/health")
async def health():
    configured = bool(
        os.getenv("POLLINATIONS_API_KEY", "").strip()
    )

    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "ok",
        "api_key_configured": configured,
    }


@app.get("/api/models")
async def models():
    """
    Return Pollinations' current model catalog.

    This endpoint itself does not expose the API key because model
    discovery is publicly available according to the API docs.
    """

    url = f"{POLLINATIONS_BASE}/v1/models"

    try:
        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
        ) as client:
            response = await client.get(url)

        if response.status_code != 200:
            return {
                "models": [],
                "source": "pollinations",
            }

        payload: Any = response.json()

    except (httpx.HTTPError, ValueError):
        return {
            "models": [],
            "source": "fallback",
        }

    raw_models = []

    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            raw_models = payload["data"]
        elif isinstance(payload.get("models"), list):
            raw_models = payload["models"]

    image_models: list[dict[str, str]] = []

    for item in raw_models:
        if not isinstance(item, dict):
            continue

        model_id = item.get("id")

        if not isinstance(model_id, str) or not model_id.strip():
            continue

        model_id = model_id.strip()

        # Only expose sensible identifiers.
        if not re.fullmatch(r"[A-Za-z0-9._/@:-]+", model_id):
            continue

        modalities = item.get("input_modalities", [])

        # If metadata clearly identifies image capability, use it.
        searchable = " ".join(
            str(x).lower()
            for x in (
                item.get("type"),
                item.get("modality"),
                item.get("output_modalities"),
                modalities,
            )
        )

        # Current catalog can differ in shape, so allow known image names
        # as a fallback.
        looks_like_image = (
            "image" in searchable
            or "image" in model_id.lower()
            or model_id.lower() in {
                "flux",
                "zimage",
                "qwen-image",
                "nanobanana-2",
            }
        )

        if not looks_like_image:
            continue

        image_models.append(
            {
                "id": model_id,
                "name": str(
                    item.get("name")
                    or item.get("title")
                    or model_id
                )[:120],
            }
        )

    # Remove duplicates.
    unique: dict[str, dict[str, str]] = {}

    for item in image_models:
        unique[item["id"]] = item

    return {
        "models": list(unique.values()),
        "source": "pollinations",
    }


@app.post("/api/generate")
async def generate_image(
    request: Request,
    payload: GenerateRequest,
):
    check_rate_limit(request)

    validate_dimensions(
        payload.width,
        payload.height,
    )

    api_key = get_api_key()

    prompt = payload.prompt

    # URL path is safely encoded with quote().
    from urllib.parse import quote

    encoded_prompt = quote(
        prompt,
        safe="",
    )

    params = {
        "model": payload.model,
        "width": payload.width,
        "height": payload.height,
        "enhance": "true" if payload.enhance else "false",
        "safe": "true" if payload.safe else "false",
        "nologo": "true",
    }

    if payload.seed >= 0:
        params["seed"] = payload.seed

    url = f"{POLLINATIONS_IMAGE_URL}/{encoded_prompt}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "image/*",
        "User-Agent": "BEHRAD-IMAGE-AI/1.0",
    }

    try:
        async with httpx.AsyncClient(
            timeout=GENERATION_TIMEOUT,
            follow_redirects=True,
        ) as client:
            response = await client.get(
                url,
                params=params,
                headers=headers,
            )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="تولید تصویر بیش از زمان مجاز طول کشید.",
        )

    except httpx.RequestError:
        raise HTTPException(
            status_code=502,
            detail="ارتباط با Pollinations برقرار نشد.",
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=safe_error_message(response.status_code),
        )

    content_type = (
        response.headers.get("content-type", "")
        .split(";")[0]
        .strip()
        .lower()
    )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
        "image/svg+xml",
    }

    if content_type not in allowed_types:
        raise HTTPException(
            status_code=502,
            detail="Pollinations پاسخ تصویری معتبر برنگرداند.",
        )

    image_bytes = response.content

    if not image_bytes:
        raise HTTPException(
            status_code=502,
            detail="تصویر خالی دریافت شد.",
        )

    # Avoid accidentally returning enormous payloads.
    max_image_bytes = 12 * 1024 * 1024

    if len(image_bytes) > max_image_bytes:
        raise HTTPException(
            status_code=502,
            detail="حجم تصویر دریافتی بیش از حد مجاز است.",
        )

    encoded = base64.b64encode(image_bytes).decode("ascii")

    data_url = f"data:{content_type};base64,{encoded}"

    return {
        "ok": True,
        "image": data_url,
        "mime": content_type,
        "model": payload.model,
        "width": payload.width,
        "height": payload.height,
        "seed": payload.seed,
        "prompt": prompt,
  }
