def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["classes"] == 113


def test_health_schema(client):
    body = client.get("/health").json()
    assert set(body.keys()) == {"status", "model_loaded", "classes"}
