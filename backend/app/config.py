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
        self.cors_origins = [
            origin.strip()
            for origin in os.environ.get("CORS_ORIGINS", "*").split(",")
            if origin.strip()
        ]


settings = Settings()
