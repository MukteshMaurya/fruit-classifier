import pytest


def _upload(path):
    return ("image", (path.name, path.open("rb"), "image/png"))


def test_predict_valid_image(client, samples_dir):
    path = samples_dir / "apple_red.png"
    response = client.post("/api/predict", files=[_upload(path)])
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "Apple Red"
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top_predictions"]) == 5
    for pred in body["top_predictions"]:
        assert set(pred.keys()) == {"class", "confidence"}
        assert 0.0 <= pred["confidence"] <= 1.0
    confs = [p["confidence"] for p in body["top_predictions"]]
    assert confs == sorted(confs, reverse=True)


def test_predict_top_class_matches_confidence(client, samples_dir):
    path = samples_dir / "banana.png"
    body = client.post("/api/predict", files=[_upload(path)]).json()
    assert body["predicted_class"] == body["top_predictions"][0]["class"]
    assert body["confidence"] == body["top_predictions"][0]["confidence"]


@pytest.mark.parametrize("name", ["banana.png", "orange.png", "strawberry.png", "tomato.png"])
def test_predict_known_fruits(client, samples_dir, name):
    expected = {
        "banana.png": "Banana",
        "orange.png": "Orange",
        "strawberry.png": "Strawberry",
        "tomato.png": "Tomato",
    }[name]
    response = client.post("/api/predict", files=[_upload(samples_dir / name)])
    assert response.status_code == 200
    assert response.json()["predicted_class"] == expected


def test_predict_missing_image(client):
    response = client.post("/api/predict")
    assert response.status_code == 422


def test_predict_invalid_image(client, samples_dir):
    path = samples_dir / "not_an_image.txt"
    response = client.post(
        "/api/predict",
        files=[("image", (path.name, path.open("rb"), "text/plain"))],
    )
    assert response.status_code == 415


def test_predict_corrupted_image(client):
    import io

    response = client.post(
        "/api/predict",
        files=[("image", ("broken.png", io.BytesIO(b"not a real png"), "image/png"))],
    )
    assert response.status_code == 400


def test_predict_empty_file(client):
    import io

    response = client.post(
        "/api/predict",
        files=[("image", ("empty.png", io.BytesIO(b""), "image/png"))],
    )
    assert response.status_code == 400


def test_predict_oversized_file(client, samples_dir):
    import io

    data = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"x" * (11 * 1024 * 1024))
    response = client.post(
        "/api/predict",
        files=[("image", ("big.png", data, "image/png"))],
    )
    assert response.status_code == 413
