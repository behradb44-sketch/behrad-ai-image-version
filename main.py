import os
import re
import base64
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# APP CONFIG
# ============================================================

APP_NAME = "BEHRAD IMAGE AI"
APP_VERSION = "2.1.0"

CLOUDFLARE_ACCOUNT_ID = os.getenv(
    "CLOUDFLARE_ACCOUNT_ID",
    ""
).strip()

CLOUDFLARE_API_TOKEN = os.getenv(
    "CLOUDFLARE_API_TOKEN",
    ""
).strip()


# ============================================================
# CLOUDFLARE AI MODELS
# ============================================================

# Image generation
IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"

# Persian -> English translation
TRANSLATION_MODEL = os.getenv(
    "CLOUDFLARE_TRANSLATION_MODEL",
    "@cf/meta/m2m100-1.2b"
).strip()


CLOUDFLARE_BASE_URL = (
    "https://api.cloudflare.com/client/v4/accounts/"
    f"{CLOUDFLARE_ACCOUNT_ID}/ai/run"
)

IMAGE_URL = (
    f"{CLOUDFLARE_BASE_URL}/{IMAGE_MODEL}"
)

TRANSLATION_URL = (
    f"{CLOUDFLARE_BASE_URL}/{TRANSLATION_MODEL}"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION
)


# ============================================================
# RATE LIMITER
# ============================================================

RATE_LIMIT_REQUESTS = 8
RATE_LIMIT_WINDOW = 60

request_log = defaultdict(deque)


# ============================================================
# REQUEST MODEL
# ============================================================

class GenerateRequest(BaseModel):

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2048
    )

    width: int = Field(
        default=768,
        ge=512,
        le=1024
    )

    height: int = Field(
        default=768,
        ge=512,
        le=1024
    )

    seed: int = Field(
        default=-1,
        ge=-1,
        le=2147483647
    )

    steps: int = Field(
        default=4,
        ge=1,
        le=8
    )

    enhance: bool = False

    safe: bool = True

    translate: bool = True


# ============================================================
# HELPERS
# ============================================================

def cloudflare_configured() -> bool:
    return bool(
        CLOUDFLARE_ACCOUNT_ID
        and CLOUDFLARE_API_TOKEN
    )


def has_persian_or_arabic(text: str) -> bool:
    """
    Detect Persian / Arabic Unicode characters.

    We only call the translation model when necessary,
    so English prompts remain as fast as possible.
    """

    return bool(
        re.search(
            r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]",
            text
        )
    )


def check_rate_limit(client_id: str) -> bool:

    now = time.monotonic()

    queue = request_log[client_id]

    while queue and (
        now - queue[0] > RATE_LIMIT_WINDOW
    ):
        queue.popleft()

    if len(queue) >= RATE_LIMIT_REQUESTS:
        return False

    queue.append(now)

    return True


def error_message(data: object) -> str:

    if isinstance(data, dict):

        errors = data.get("errors")

        if isinstance(errors, list) and errors:

            messages = []

            for item in errors:

                if isinstance(item, dict):

                    message = (
                        item.get("message")
                        or item.get("code")
                    )

                    if message:
                        messages.append(str(message))

                elif item:

                    messages.append(str(item))

            if messages:
                return " | ".join(messages)

        messages = data.get("messages")

        if isinstance(messages, list) and messages:

            values = []

            for item in messages:

                if isinstance(item, dict):

                    message = (
                        item.get("message")
                        or item.get("code")
                    )

                    if message:
                        values.append(str(message))

                elif item:

                    values.append(str(item))

            if values:
                return " | ".join(values)

    return "Cloudflare AI request failed."


# ============================================================
# CLOUDFLARE REQUEST
# ============================================================

async def cloudflare_post(
    client: httpx.AsyncClient,
    url: str,
    body: dict
) -> tuple[int, dict]:

    response = await client.post(

        url,

        headers={
            "Authorization":
                f"Bearer {CLOUDFLARE_API_TOKEN}",

            "Content-Type":
                "application/json",
        },

        json=body,
    )

    try:

        data = response.json()

    except Exception:

        data = {
            "raw": response.text[:2000]
        }

    return response.status_code, data


# ============================================================
# PERSIAN -> ENGLISH TRANSLATION
# ============================================================

async def translate_persian_to_english(
    client: httpx.AsyncClient,
    prompt: str
) -> str:

    body = {

        "text": prompt,

        "source_lang": "fa",

        "target_lang": "en",
    }

    status, data = await cloudflare_post(
        client,
        TRANSLATION_URL,
        body
    )

    # --------------------------------------------------------
    # Authentication
    # --------------------------------------------------------

    if status == 401:

        raise HTTPException(
            status_code=502,
            detail=(
                "Cloudflare API token is invalid "
                "or unauthorized."
            )
        )

    # --------------------------------------------------------
    # Permission
    # --------------------------------------------------------

    if status == 403:

        raise HTTPException(
            status_code=502,
            detail=(
                "Cloudflare API token does not have "
                "permission to run Workers AI."
            )
        )

    # --------------------------------------------------------
    # Rate / free allowance
    # --------------------------------------------------------

    if status == 429:

        raise HTTPException(
            status_code=429,
            detail=(
                "Cloudflare free AI allowance or "
                "rate limit was reached. "
                "Please try again later."
            )
        )

    # --------------------------------------------------------
    # Other errors
    # --------------------------------------------------------

    if status >= 400:

        raise HTTPException(
            status_code=502,
            detail=(
                "Translation failed: "
                f"{error_message(data)}"
            )
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = (
        data.get("result")
        if isinstance(data, dict)
        else None
    )

    translated = None

    if isinstance(result, dict):

        translated = result.get(
            "translated_text"
        )

    if (
        not isinstance(translated, str)
        or not translated.strip()
    ):

        raise HTTPException(
            status_code=502,
            detail=(
                "Translation service returned "
                "an empty result."
            )
        )

    return translated.strip()


# ============================================================
# IMAGE RESULT
# ============================================================

def decode_image_result(
    data: dict
) -> tuple[str, str]:

    result = data.get("result")

    if not isinstance(result, dict):

        raise HTTPException(
            status_code=502,
            detail=(
                "Cloudflare returned an "
                "unexpected image response."
            )
        )

    image_b64 = result.get("image")

    if (
        not isinstance(image_b64, str)
        or not image_b64.strip()
    ):

        raise HTTPException(
            status_code=502,
            detail=(
                "Cloudflare did not return an image."
            )
        )

    image_b64 = image_b64.strip()

    # --------------------------------------------------------
    # Remove data URL prefix if present
    # --------------------------------------------------------

    if image_b64.startswith("data:"):

        try:

            image_b64 = image_b64.split(
                ",",
                1
            )[1]

        except IndexError:

            raise HTTPException(
                status_code=502,
                detail=(
                    "Invalid image data returned "
                    "by Cloudflare."
                )
            )

    # --------------------------------------------------------
    # Validate Base64
    # --------------------------------------------------------

    try:

        raw = base64.b64decode(
            image_b64,
            validate=True
        )

    except Exception:

        raise HTTPException(
            status_code=502,
            detail=(
                "Cloudflare returned invalid "
                "base64 image data."
            )
        )

    # --------------------------------------------------------
    # Image size protection
    # --------------------------------------------------------

    if len(raw) > 15 * 1024 * 1024:

        raise HTTPException(
            status_code=502,
            detail=(
                "Generated image is too large."
            )
        )

    # FLUX normally returns PNG.
    return (
        "data:image/png;base64,"
        + image_b64,
        image_b64
    )


# ============================================================
# SECURITY HEADERS
# ============================================================

@app.middleware("http")
async def security_headers(
    request: Request,
    call_next
):

    response = await call_next(request)

    response.headers[
        "X-Content-Type-Options"
    ] = "nosniff"

    response.headers[
        "X-Frame-Options"
    ] = "SAMEORIGIN"

    response.headers[
        "Referrer-Policy"
    ] = "strict-origin-when-cross-origin"

    response.headers[
        "Permissions-Policy"
    ] = (
        "camera=(), "
        "microphone=(), "
        "geolocation=()"
    )

    return response


# ============================================================
# REQUEST SIZE LIMIT
# ============================================================

@app.middleware("http")
async def request_size_limit(
    request: Request,
    call_next
):

    content_length = request.headers.get(
        "content-length"
    )

    if content_length:

        try:

            if int(content_length) > 32 * 1024:

                return JSONResponse(
                    status_code=413,
                    content={
                        "success": False,
                        "detail": (
                            "Request body is too large."
                        )
                    }
                )

        except ValueError:

            pass

    return await call_next(request)


# ============================================================
# HOME
# ============================================================

@app.get("/")
async def home():

    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "detail": (
                    "Frontend file not found."
                )
            }
        )

    return FileResponse(index_file)


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
async def health():

    return {

        "service": APP_NAME,

        "version": APP_VERSION,

        "status": "ok",

        "provider":
            "Cloudflare Workers AI",

        "image_model":
            IMAGE_MODEL,

        "translation_model":
            TRANSLATION_MODEL,

        "cloudflare_configured":
            cloudflare_configured(),
    }


# ============================================================
# MODELS
# ============================================================

@app.get("/api/models")
async def models():

    return {

        "success": True,

        "models": [

            {
                "id": IMAGE_MODEL,

                "name":
                    "FLUX.1 Schnell",

                "type":
                    "text-to-image",

                "provider":
                    "Cloudflare Workers AI",
            }

        ],

        "translation": {

            "enabled": True,

            "model":
                TRANSLATION_MODEL,

            "source":
                "Persian",

            "target":
                "English",
        }
    }


# ============================================================
# GENERATE IMAGE
# ============================================================

@app.post("/api/generate")
async def generate(
    payload: GenerateRequest,
    request: Request
):

    # --------------------------------------------------------
    # Cloudflare configuration
    # --------------------------------------------------------

    if not cloudflare_configured():

        raise HTTPException(
            status_code=500,
            detail=(
                "Cloudflare is not configured "
                "on the server."
            )
        )

    # --------------------------------------------------------
    # Rate limit
    # --------------------------------------------------------

    client_host = (
        request.client.host
        if request.client
        else "unknown"
    )

    if not check_rate_limit(client_host):

        raise HTTPException(
            status_code=429,
            detail=(
                "Too many requests. "
                "Please wait a little and try again."
            )
        )

    # --------------------------------------------------------
    # Original prompt
    # --------------------------------------------------------

    original_prompt = payload.prompt.strip()

    if not original_prompt:

        raise HTTPException(
            status_code=422,
            detail="Prompt cannot be empty."
        )

    # --------------------------------------------------------
    # Defaults
    # --------------------------------------------------------

    final_prompt = original_prompt

    translated = False

    timeout = httpx.Timeout(

        connect=20.0,

        read=180.0,

        write=30.0,

        pool=20.0
    )

    # --------------------------------------------------------
    # Cloudflare
    # --------------------------------------------------------

    async with httpx.AsyncClient(
        timeout=timeout
    ) as client:

        # ====================================================
        # AUTOMATIC PERSIAN DETECTION
        # ====================================================

        if (
            payload.translate
            and has_persian_or_arabic(
                original_prompt
            )
        ):

            final_prompt = (
                await translate_persian_to_english(
                    client,
                    original_prompt
                )
            )

            translated = True

        # ====================================================
        # IMAGE REQUEST
        # ====================================================

        body = {

            "prompt":
                final_prompt,

            "steps":
                payload.steps,
        }

        # ----------------------------------------------------
        # Seed
        # ----------------------------------------------------

        if payload.seed != -1:

            body["seed"] = payload.seed

        # ----------------------------------------------------
        # FLUX
        # ----------------------------------------------------

        status, data = await cloudflare_post(

            client,

            IMAGE_URL,

            body
        )

    # ========================================================
    # IMAGE ERROR HANDLING
    # ========================================================

    if status == 401:

        raise HTTPException(
            status_code=502,
            detail=(
                "Cloudflare API token is invalid "
                "or unauthorized."
            )
        )

    if status == 403:

        raise HTTPException(
            status_code=502,
            detail=(
                "Cloudflare API token does not have "
                "permission to run Workers AI."
            )
        )

    if status == 429:

        raise HTTPException(
            status_code=429,
            detail=(
                "Cloudflare free AI allowance or "
                "rate limit was reached. "
                "Please try again later."
            )
        )

    if status == 400:

        raise HTTPException(
            status_code=400,
            detail=(
                "Cloudflare rejected the image request: "
                f"{error_message(data)}"
            )
        )

    if status >= 400:

        raise HTTPException(
            status_code=502,
            detail=(
                "Image generation failed: "
                f"{error_message(data)}"
            )
        )

    # ========================================================
    # IMAGE
    # ========================================================

    image_url, _ = decode_image_result(data)

    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "success": True,

        "image":
            image_url,

        "model":
            IMAGE_MODEL,

        # Original user prompt
        "prompt":
            original_prompt,

        # English prompt actually sent to FLUX
        "translated_prompt":
            final_prompt
            if translated
            else None,

        "translated":
            translated,

        "seed":
            payload.seed,

        "steps":
            payload.steps,

        "width":
            payload.width,

        "height":
            payload.height,

        "enhance":
            payload.enhance,

        "safe":
            payload.safe,
    }


# ============================================================
# STATIC FILES
# ============================================================

if STATIC_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(
            directory=STATIC_DIR
        ),
        name="static"
    )


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
                        )
