import copy

import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.api import rerank as rerank_api
from app.services.reranker_service import RerankResult


class FakeRegistry:
    model_states = {
        "embeddings": {
            "enabled": True,
            "loaded": True,
            "model_id": "fake-embedding",
            "revision": None,
            "dimensions": 384,
        },
        "reranker": {
            "enabled": False,
            "loaded": False,
            "model_id": "fake-reranker",
            "revision": None,
        },
    }

    @classmethod
    def from_yaml(cls, path):
        return cls()

    def load_enabled(self):
        return None

    def readiness(self):
        return copy.deepcopy(self.model_states)

    def is_available(self, name):
        state = self.model_states[name]
        return state["enabled"] and state["loaded"]

    def unavailable_reason(self, name):
        return f"{name} unavailable"

    def all_enabled_models_available(self):
        return all(
            not state["enabled"] or state["loaded"]
            for state in self.model_states.values()
        )


@pytest.fixture(autouse=True)
def fake_model_registry(monkeypatch):
    monkeypatch.setattr(
        FakeRegistry,
        "model_states",
        {
            "embeddings": {
                "enabled": True,
                "loaded": True,
                "model_id": "fake-embedding",
                "revision": None,
                "dimensions": 384,
            },
            "reranker": {
                "enabled": False,
                "loaded": False,
                "model_id": "fake-reranker",
                "revision": None,
            },
        },
    )
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
        "embeddings": {
            "enabled": True,
            "loaded": True,
            "model_id": "fake-embedding",
            "revision": None,
            "dimensions": 384,
        },
        "reranker": {
            "enabled": False,
            "loaded": False,
            "model_id": "fake-reranker",
            "revision": None,
        },
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


def test_rerank_returns_503_when_model_is_disabled():
    with TestClient(app_main.app) as client:
        response = client.post(
            "/v1/rerank",
            json={
                "query": "music",
                "candidates": [{"id": "event_1", "text": "jazz"}],
            },
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "reranker unavailable"}


def test_embedding_text_returns_503_when_model_is_unavailable():
    FakeRegistry.model_states["embeddings"]["loaded"] = False

    with TestClient(app_main.app) as client:
        response = client.post("/v1/embeddings/text", json={"text": "music"})

    assert response.status_code == 503
    assert response.json() == {"detail": "embeddings unavailable"}


def test_embedding_batch_returns_503_when_model_is_unavailable():
    FakeRegistry.model_states["embeddings"]["loaded"] = False

    with TestClient(app_main.app) as client:
        response = client.post("/v1/embeddings/batch", json={"texts": ["music"]})

    assert response.status_code == 503
    assert response.json() == {"detail": "embeddings unavailable"}


def test_rerank_reaches_service_when_enabled_and_available(monkeypatch):
    FakeRegistry.model_states["reranker"].update({"enabled": True, "loaded": True})
    calls = []

    class FakeRerankerService:
        def rerank(self, **kwargs):
            calls.append(kwargs)
            return RerankResult(
                model="fake-reranker",
                results=[
                    {
                        "id": "event_1",
                        "score": 0.9,
                        "rank": 1,
                        "metadata": {"source": "test"},
                    }
                ],
            )

    monkeypatch.setattr(rerank_api, "_service", lambda request: FakeRerankerService())

    with TestClient(app_main.app) as client:
        response = client.post(
            "/v1/rerank",
            json={
                "query": "music",
                "candidates": [
                    {"id": "event_1", "text": "jazz", "metadata": {"source": "test"}}
                ],
            },
        )

    assert response.status_code == 200
    assert response.json()["model"] == "fake-reranker"
    assert calls == [
        {
            "query": "music",
            "candidates": [
                {"id": "event_1", "text": "jazz", "metadata": {"source": "test"}}
            ],
            "top_k": None,
        }
    ]


def test_rerank_returns_503_when_enabled_model_failed_to_load():
    FakeRegistry.model_states["reranker"].update({"enabled": True, "loaded": False})

    with TestClient(app_main.app) as client:
        response = client.post(
            "/v1/rerank",
            json={
                "query": "music",
                "candidates": [{"id": "event_1", "text": "jazz"}],
            },
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "reranker unavailable"}


def test_ready_returns_503_when_enabled_model_failed_to_load():
    FakeRegistry.model_states["embeddings"]["loaded"] = False

    with TestClient(app_main.app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["embeddings"]["loaded"] is False
