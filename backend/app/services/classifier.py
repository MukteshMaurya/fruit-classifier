import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ModelNotLoadedError(RuntimeError):
    pass


class FruitClassifier:
    """Fruit image classifier backed by an ONNX Runtime model.

    Preprocessing (matches the model's training pipeline):
    RGB -> resize 224x224 bicubic -> rescale [0,1] ->
    ImageNet normalization -> CHW -> batch dim.
    """

    def __init__(self, model_path: str, labels_path: str):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.labels: dict[int, str] = {}
        self.session: ort.InferenceSession | None = None
        self.input_name = "pixel_values"
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists() or not self.labels_path.exists():
            return
        with open(self.labels_path, encoding="utf-8") as f:
            raw = json.load(f)
        self.labels = {int(k): v for k, v in raw.items()}
        options = ort.SessionOptions()
        options.intra_op_num_threads = 0  # let ORT choose
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    def preprocess(self, image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        resized = rgb.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        arr = np.transpose(arr, (2, 0, 1))
        return np.ascontiguousarray(arr[np.newaxis, :], dtype=np.float32)

    def predict(self, image: Image.Image, top_k: int = 5) -> list[dict]:
        if self.session is None:
            raise ModelNotLoadedError("Model is not loaded")
        x = self.preprocess(image)
        logits = self.session.run(None, {self.input_name: x})[0][0]
        probs = self._softmax(logits)
        top_idx = np.argsort(-probs)[:top_k]
        return [
            {
                "class": self.labels.get(int(i), "unknown"),
                "confidence": float(probs[i]),
            }
            for i in top_idx
        ]

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / exp.sum()
