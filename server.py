from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uvicorn
from dotenv import load_dotenv
from utils.analyzer_functions import analyze_food_image, analyze_food_text

load_dotenv(dotenv_path=".env")


def _parse_csv_env(name: str) -> list[str]:
    value = os.getenv(name, "")
    return [item.strip() for item in value.split(",") if item.strip()]


# Local Vite dev/preview ports plus Beet Health deployed web apps on justbots.tech.
_cors_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]
_cors_origins = list(dict.fromkeys(_cors_origins + _parse_csv_env("CORS_ORIGINS")))
_cors_origin_regex = os.getenv(
    "CORS_ORIGIN_REGEX",
    r"^https://beethealth(?:-[a-z0-9-]+)?\.justbots\.tech$",
)

app = FastAPI(title="Beet Food-Image Analyzer", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=_cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MEDIA_TYPE_MAP = {
    "jpg":  "image/jpeg",
    "jpeg": "image/jpeg",
    "png":  "image/png",
    "webp": "image/webp",
}


@app.get("/health")
# Check if server is running
def health():
    return {"status": "ok"}

@app.post("/analyze_photo")
async def analyze_photo(file: UploadFile = File(...), ai_provider = "openai", model: str = None):
    """
    Upload a food image and receive a JSON breakdown of identified dishes and their macros.

    Response shape:
    {
        "dishes": {
            "<dish_name>": {
                "kcal": float,
                "protein": float,   // grams
                "fat": float,       // grams
                "carbs": float,     // grams
                "fiber": float,     // grams
                "sugar": float,     // grams
                "quantity": str,    // e.g. "1 katori ~200g"
                "confidence": float // 0–1
            },
            ...
        },
        "total": { "kcal": ..., "protein": ..., "fat": ..., "carbs": ..., "fiber": ..., "sugar": ... },
        "overall_confidence": float,
        "notes": str   // optional
    }
    """
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    media_type = MEDIA_TYPE_MAP.get(ext)
    if not media_type:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Accepted: jpg, jpeg, png, webp.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = analyze_food_image(image_bytes, media_type, ai_provider, model=model)

    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    except RuntimeError as e:
        msg = str(e)
        transient = any(k in msg for k in ("rate limit", "timed out", "connect"))
        status = 503 if transient else 500
        raise HTTPException(status_code=status, detail=msg)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return JSONResponse(content=result)

@app.post("/analyze_text")
async def analyze_text(query: str, ai_provider: str = "openai", model: str | None = None):
    """
    Send a text description of food and receive a JSON breakdown of identified dishes and their macros.
    Same response shape as /analyze_photo.
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = analyze_food_text(query.strip(), ai_provider, model=model)

    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except RuntimeError as e:
        msg = str(e)
        transient = any(k in msg for k in ("rate limit", "timed out", "connect"))
        status = 503 if transient else 500
        raise HTTPException(status_code=status, detail=msg)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run("server:app", reload=True)
