"""Opt-in reranker contracts; requires both real-model environment switches."""

from __future__ import annotations

import math
import os

import pytest
from fastapi.testclient import TestClient

from app import main as app_main


pytestmark = [
    pytest.mark.reranker,
    pytest.mark.skipif(
        os.getenv("RUN_REAL_MODEL_TESTS", "").lower() != "true",
        reason="set RUN_REAL_MODEL_TESTS=true and RUN_RERANKER_TESTS=true to test the real reranker",
    ),
    pytest.mark.skipif(
        os.getenv("RUN_RERANKER_TESTS", "").lower() != "true",
        reason="set RUN_RERANKER_TESTS=true to enable opt-in real reranker tests",
    ),
]

TEST_TOKEN = "real-reranker-contract-test-token"


@pytest.fixture(scope="module")
def real_reranker_client():
    previous = {
        key: os.environ.get(key)
        for key in ("LOOPIN_SERVICE_TOKEN", "LOOPIN_EMBEDDINGS_ENABLED", "LOOPIN_RERANKER_ENABLED")
    }
    os.environ["LOOPIN_SERVICE_TOKEN"] = TEST_TOKEN
    os.environ["LOOPIN_EMBEDDINGS_ENABLED"] = "true"
    os.environ["LOOPIN_RERANKER_ENABLED"] = "true"
    try:
        with TestClient(app_main.app) as client:
            assert client.app.state.models.is_available("reranker"), (
                "Configured reranker did not load; inspect the model-load logs or cache/network access. "
                f"State: {client.app.state.models.readiness()['reranker']}"
            )
            yield client
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_real_reranker_http_contract(real_reranker_client):
    candidates = [
        {"id": "relevant", "text": "Live jazz concert with a small audience", "metadata": {"kind": "music"}},
        {"id": "unrelated", "text": "Hands-on Python web programming workshop", "metadata": {"kind": "technology"}},
        {"id": "overlap", "text": "Lecture about the history of jazz recordings", "metadata": {"kind": "talk"}},
    ]
    response = real_reranker_client.post(
        "/v1/rerank",
        json={"query": "I want a live jazz concert", "candidates": candidates, "top_k": 2},
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"model", "results"}
    assert body["model"] == real_reranker_client.app.state.models.reranker_config["model_id"]
    assert len(body["results"]) == 2
    assert body["results"][0]["id"] == "relevant"
    assert [item["rank"] for item in body["results"]] == [1, 2]
    assert all(isinstance(item["score"], (int, float)) and math.isfinite(item["score"]) for item in body["results"])
    assert body["results"][0]["score"] >= body["results"][1]["score"]
    supplied_metadata = {candidate["id"]: candidate["metadata"] for candidate in candidates}
    assert all(item["metadata"] == supplied_metadata[item["id"]] for item in body["results"])
