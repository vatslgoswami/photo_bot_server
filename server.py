from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
from utils.analyzer_functions import analyze_food_image

load_dotenv(dotenv_path=".env")

app = FastAPI(title="Beet Food-Image Analyzer", version="1.0.0")

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
async def analyze(file: UploadFile = File(...), ai_provider = "openai", model: str = None):
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

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
