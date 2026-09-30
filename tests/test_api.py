from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_empty_query():
    response = client.post(
        "/chat",
        json={"query": ""},
    )

    assert response.status_code == 422