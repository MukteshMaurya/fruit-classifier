import io
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, Request
from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.services.classifier import ModelNotLoadedError

router = APIRouter()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def _validate_image(file: UploadFile, data: bytes) -> Image.Image:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image format '{suffix or 'none'}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
        image = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Corrupted or invalid image")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")
    return image


@router.get("/health")
def health(request: Request):
    classifier = request.app.state.classifier
    return {
        "status": "ok",
        "model_loaded": classifier.is_loaded,
        "classes": len(classifier.labels),
    }


@router.post("/api/predict")
async def predict(request: Request, image: UploadFile = File(...)):
    classifier = request.app.state.classifier
    if not classifier.is_loaded:
        raise HTTPException(
            status_code=503, detail="Model is not available"
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = await image.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large. Maximum size is {settings.max_upload_mb} MB",
        )

    pil_image = _validate_image(image, data)

    try:
        results = classifier.predict(pil_image, top_k=settings.top_k)
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="Model is not available")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")

    if not results:
        raise HTTPException(status_code=500, detail="Prediction failed")

    top = results[0]
    return {
        "predicted_class": top["class"],
        "confidence": top["confidence"],
        "top_predictions": results,
    }
