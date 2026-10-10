import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_ASSETS_DIR = BASE_DIR / "model_assets"


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
        self.rejection_confidence = float(
            os.environ.get("REJECTION_CONFIDENCE", "0.45")
        )
        self.rejection_margin = float(
            os.environ.get("REJECTION_MARGIN", "0.30")
        )
        self.cors_origins = [
            origin.strip()
            for origin in os.environ.get("CORS_ORIGINS", "*").split(",")
            if origin.strip()
        ]


settings = Settings()
