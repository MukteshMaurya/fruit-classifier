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
        """Top-k predictions, highest confidence first (existing API)."""
        return self.predict_detailed(image, top_k=top_k)["top_predictions"]

    def predict_detailed(self, image: Image.Image, top_k: int = 5) -> dict:
        """Prediction result plus the signals used for non-fruit rejection.

        Returns a dict with:
          top_class, confidence            — top-1 class and its probability
          second_class, second_confidence  — runner-up class and probability
          margin                           — confidence gap (top1 - top2)
          top_predictions                  — top-k list, same as predict()
        """
        if self.session is None:
            raise ModelNotLoadedError("Model is not loaded")
        x = self.preprocess(image)
        logits = self.session.run(None, {self.input_name: x})[0][0]
        probs = self._softmax(logits)
        order = np.argsort(-probs)
        top_i = int(order[0])
        second_i = int(order[1])
        top_predictions = [
            {
                "class": self.labels.get(int(i), "unknown"),
                "confidence": float(probs[i]),
            }
            for i in order[:top_k]
        ]
        return {
            "top_class": self.labels.get(top_i, "unknown"),
            "confidence": float(probs[top_i]),
            "second_class": self.labels.get(second_i, "unknown"),
            "second_confidence": float(probs[second_i]),
            "margin": float(probs[top_i] - probs[second_i]),
            "top_predictions": top_predictions,
        }

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / exp.sum()
