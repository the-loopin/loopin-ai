import pytest
from fastapi.testclient import TestClient

from app import main as app_main


class FakeRegistry:
    @classmethod
    def from_yaml(cls, path):
        return cls()

    def load_all(self):
        return None

    def readiness(self):
        return {
            "embedding_model": "fake-embedding",
            "reranker_model": "fake-reranker",
        }


@pytest.fixture(autouse=True)
def fake_model_registry(monkeypatch):
    monkeypatch.setattr(app_main, "ModelRegistry", FakeRegistry)


def test_health_and_ready_with_mocked_registry():
    with TestClient(app_main.app) as client:
        health_response = client.get("/health")
        ready_response = client.get("/ready")

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}
    assert ready_response.status_code == 200
    assert ready_response.json() == {
        "status": "ready",
        "embedding_model": "fake-embedding",
        "reranker_model": "fake-reranker",
    }


def test_embedding_batch_rejects_too_many_items():
    with TestClient(app_main.app) as client:
        response = client.post(
            "/v1/embeddings/batch",
            json={"texts": ["music"] * 129, "input_type": "passage"},
        )

    assert response.status_code == 422


def test_embedding_batch_rejects_text_over_limit():
    with TestClient(app_main.app) as client:
        response = client.post(
            "/v1/embeddings/batch",
            json={"texts": ["x" * 4097], "input_type": "passage"},
        )

    assert response.status_code == 422


def test_rerank_rejects_too_many_candidates():
    with TestClient(app_main.app) as client:
        response = client.post(
            "/v1/rerank",
            json={
                "query": "music",
                "candidates": [
                    {"id": f"event_{index}", "text": "jazz"}
                    for index in range(101)
                ],
            },
        )

    assert response.status_code == 422


def test_rerank_rejects_top_k_over_limit():
    with TestClient(app_main.app) as client:
        response = client.post(
            "/v1/rerank",
            json={
                "query": "music",
                "top_k": 51,
                "candidates": [{"id": "event_1", "text": "jazz"}],
            },
        )

    assert response.status_code == 422
