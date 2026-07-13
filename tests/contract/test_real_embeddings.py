"""Opt-in contracts against the configured Hugging Face embedding model.

These tests intentionally use the application lifespan so endpoint authentication and the
production response models are exercised. They are skipped before model construction unless
RUN_REAL_MODEL_TESTS=true is supplied.
"""

from __future__ import annotations

import math
import os

import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.services.embedding_service import EmbeddingService


pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        os.getenv("RUN_REAL_MODEL_TESTS", "").lower() != "true",
        reason="set RUN_REAL_MODEL_TESTS=true to load and test the real embedding model",
    ),
]

TEST_TOKEN = "real-model-contract-test-token"


@pytest.fixture(scope="module")
def real_embedding_client():
    previous = {
        key: os.environ.get(key)
        for key in ("LOOPIN_SERVICE_TOKEN", "LOOPIN_EMBEDDINGS_ENABLED", "LOOPIN_RERANKER_ENABLED")
    }
    os.environ["LOOPIN_SERVICE_TOKEN"] = TEST_TOKEN
    os.environ["LOOPIN_EMBEDDINGS_ENABLED"] = "true"
    os.environ["LOOPIN_RERANKER_ENABLED"] = "false"
    try:
        with TestClient(app_main.app) as client:
            assert client.app.state.models.is_available("embeddings"), (
                "Configured embedding model did not load; inspect the model-load logs or cache/network access. "
                f"State: {client.app.state.models.readiness()['embeddings']}"
            )
            yield client
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _assert_vector(vector: list[float], dimensions: int) -> None:
    assert len(vector) == dimensions
    assert all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector)
    assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0, rel_tol=1e-4, abs_tol=1e-4)


def test_real_embedding_service_contract(real_embedding_client):
    registry = real_embedding_client.app.state.models
    result = EmbeddingService(registry).embed(["Türkçe canlı müzik etkinliği"], input_type="query")
    assert result.model == registry.embedding_config["model_id"]
    assert result.dimensions == 384
    assert len(result.embeddings) == 1
    _assert_vector(result.embeddings[0], result.dimensions)


def test_real_embedding_http_contract_and_authentication(real_embedding_client):
    headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
    single = real_embedding_client.post(
        "/v1/embeddings/text",
        json={"text": "Bakıda canlı caz tədbiri", "input_type": "query"},
        headers=headers,
    )
    assert single.status_code == 200
    body = single.json()
    config = real_embedding_client.app.state.models.embedding_config
    assert set(body) == {"model", "dimensions", "embedding"}
    assert body["model"] == config["model_id"]
    assert body["dimensions"] == config["dimensions"] == 384
    _assert_vector(body["embedding"], body["dimensions"])

    batch = real_embedding_client.post(
        "/v1/embeddings/batch",
        json={
            "texts": [
                "Bakıda açıq havada canlı caz gecəsi",
                "İstanbul'da canlı caz konseri",
                "Live jazz night on an open-air terrace",
            ],
            "input_type": "passage",
        },
        headers=headers,
    )
    assert batch.status_code == 200
    batch_body = batch.json()
    assert set(batch_body) == {"model", "dimensions", "embeddings"}
    assert batch_body["model"] == config["model_id"]
    assert isinstance(batch_body["embeddings"], list) and len(batch_body["embeddings"]) == 3
    for vector in batch_body["embeddings"]:
        _assert_vector(vector, 384)

    unauthenticated = real_embedding_client.post(
        "/v1/embeddings/text", json={"text": "authentication contract"}
    )
    assert unauthenticated.status_code == 401
    assert unauthenticated.json() == {"detail": "Invalid service authentication."}
