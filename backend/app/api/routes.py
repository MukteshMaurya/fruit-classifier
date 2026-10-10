import io
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, Request
from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.services.classifier import ModelNotLoadedError

router = APIRouter()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

NON_FRUIT_MESSAGE = (
    "Non-fruit image detected. Please upload an image of a "
    "supported fruit."
)


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
        detail = classifier.predict_detailed(pil_image, top_k=settings.top_k)
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="Model is not available")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")

    if not detail["top_predictions"]:
        raise HTTPException(status_code=500, detail="Prediction failed")

    # Closed-set models force every input into one of the fruit
    # classes, so a low confidence alone is not a reliable
    # signal. Reject when the confidence OR the prediction
    # margin falls below thresholds derived from a validation
    # set of supported fruit images and non-fruit images
    # (see OPENCODE_PROGRESS.md). This keeps valid fruit
    # images (even low-confidence ones, e.g. poor lighting)
    # accepted while rejecting non-fruit images.
    is_non_fruit = (
        detail["confidence"] < settings.rejection_confidence
        or detail["margin"] < settings.rejection_margin
    )
    if is_non_fruit:
        return {
            "predicted_class": None,
            "confidence": None,
            "rejected": True,
            "reason": "non_fruit",
            "message": NON_FRUIT_MESSAGE,
            "top_predictions": [],
        }

    top = detail["top_predictions"][0]
    return {
        "predicted_class": top["class"],
        "confidence": top["confidence"],
        "top_predictions": detail["top_predictions"],
    }
