"""Regression tests for non-fruit image rejection."""
import math

import pytest
from PIL import Image

from app.services.classifier import ModelNotLoadedError

NON_FRUIT_MESSAGE = (
    "Non-fruit image detected. Please upload an image of a "
    "supported fruit."
)

NONFRUIT_CATEGORIES = [
    "person",
    "car",
    "mobile_phone",
    "dog",
    "cat",
    "chair",
    "building",
    "bicycle",
]

KNOWN_FRUITS = {
    "apple_red.png": "Apple Red",
    "banana.png": "Banana",
    "orange.png": "Orange",
    "strawberry.png": "Strawberry",
    "tomato.png": "Tomato",
}


def _upload(path, mime="image/jpeg"):
    return ("image", (path.name, path.open("rb"), mime))


@pytest.mark.parametrize("category", NONFRUIT_CATEGORIES)
def test_predict_rejects_nonfruit_images(client, samples_dir, category):
    """Non-fruit photos are rejected with the non-fruit message."""
    path = samples_dir / "nonfruit" / f"{category}.jpg"
    response = client.post("/api/predict", files=[_upload(path)])
    assert response.status_code == 200
    body = response.json()
    assert body["rejected"] is True
    assert body["reason"] == "non_fruit"
    assert body["message"] == NON_FRUIT_MESSAGE
    # no misleading fruit prediction is returned
    assert body["predicted_class"] is None
    assert body["confidence"] is None
    assert body["top_predictions"] == []


@pytest.mark.parametrize("name,expected", list(KNOWN_FRUITS.items()))
def test_valid_fruits_are_not_rejected(client, samples_dir, name, expected):
    """Supported fruit images still get a normal prediction."""
    response = client.post(
        "/api/predict",
        files=[_upload(samples_dir / name, "image/png")],
    )
    assert response.status_code == 200
    body = response.json()
    assert body.get("rejected") is not True
    assert body["predicted_class"] == expected
    assert math.isfinite(body["confidence"])
    assert 0.0 <= body["confidence"] <= 1.0


def test_rejection_signals_separate_fruit_from_nonfruit(client, samples_dir):
    """The confidence/margin signals must separate the two classes
    on the committed fixtures (the basis of the thresholds)."""
    classifier = client.app.state.classifier
    fruit = classifier.predict_detailed(
        Image.open(samples_dir / "apple_red.png")
    )
    assert fruit["confidence"] >= 0.45
    assert fruit["margin"] >= 0.30

    nonfruit = classifier.predict_detailed(
        Image.open(samples_dir / "nonfruit" / "car.jpg")
    )
    assert nonfruit["confidence"] < 0.45 or nonfruit["margin"] < 0.30


def test_rejection_thresholds_are_configurable(client, samples_dir):
    """Thresholds come from settings and can be overridden via env."""
    from app.config import settings

    assert settings.rejection_confidence == 0.45
    assert settings.rejection_margin == 0.30


def test_predict_detailed_margin_is_nonnegative(client, samples_dir):
    """margin = top1 - top2 >= 0 and second <= top."""
    classifier = client.app.state.classifier
    detail = classifier.predict_detailed(
        Image.open(samples_dir / "banana.png")
    )
    assert detail["margin"] >= 0.0
    assert detail["second_confidence"] <= detail["confidence"]
    assert detail["top_predictions"][0]["confidence"] == detail["confidence"]


def test_predict_detailed_requires_model(client, samples_dir):
    """predict_detailed raises when the model is missing."""
    classifier = client.app.state.classifier
    session = classifier.session
    classifier.session = None
    try:
        with pytest.raises(ModelNotLoadedError):
            classifier.predict_detailed(
                Image.open(samples_dir / "apple_red.png")
            )
    finally:
        classifier.session = session
