import logging
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_ASSETS_DIR = BASE_DIR / "model_assets"

logger = logging.getLogger(__name__)


def _parse_threshold(name: str, default: str) -> float:
    """Parse a rejection threshold from the environment.

    Accepts either a fraction ("0.45") or a percentage
    ("45"). Values above 1 are treated as percentages and
    divided by 100, so a percentage-style value cannot
    make the threshold exceed 1.0 and reject every image.
    Invalid values fall back to the default.
    """
    raw = os.environ.get(name, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid value %r for %s; using default %s",
            raw, name, default,
        )
        return float(default)
    if value > 1.0:
        value = value / 100.0
    return min(1.0, max(0.0, value))


class Settings:
    """Application configuration from environment variables."""

    def __init__(self):
        self.model_path = os.environ.get(
            "MODEL_PATH", str(MODEL_ASSETS_DIR / "model_int8.onnx")
        )
        self.labels_path = os.environ.get(
            "LABELS_PATH", str(MODEL_ASSETS_DIR / "labels.json")
        )
        self.max_upload_mb = int(os.environ.get("MAX_UPLOAD_MB", "10"))
        self.top_k = int(os.environ.get("TOP_K", "5"))
        # Non-fruit rejection thresholds, derived from a validation set
        # (565 Fruits-360 test images vs 100 non-fruit photos):
        # fruit min confidence 0.573 / min margin 0.358,
        # non-fruit max confidence 0.332 / max margin 0.264.
        # Accepts fractions ("0.45") or percentages ("45").
        self.rejection_confidence = _parse_threshold(
            "REJECTION_CONFIDENCE", "0.45"
        )
        self.rejection_margin = _parse_threshold(
            "REJECTION_MARGIN", "0.30"
        )
        logger.info(
            "Non-fruit rejection thresholds: confidence>=%s margin>=%s",
            self.rejection_confidence, self.rejection_margin,
        )
        self.cors_origins = [
            origin.strip()
            for origin in os.environ.get("CORS_ORIGINS", "*").split(",")
            if origin.strip()
        ]


settings = Settings()
