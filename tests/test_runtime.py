import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.api import embeddings as embeddings_api
from app.runtime import InferenceLimiter, InferenceQueueFull, InferenceRuntimeSettings


class FakeRegistry:
    @classmethod
    def from_yaml(cls, path):
        return cls()

    def load_enabled(self):
        return None

    def is_available(self, name):
        return name == "embeddings"

    def unavailable_reason(self, name):
        return f"{name} unavailable"

    def readiness_status(self):
        return "ready"

    def readiness(self):
        return {
            "embeddings": {
                "enabled": True,
                "loaded": True,
                "model_id": "fake-embedding",
                "revision": None,
                "dimensions": 1,
            },
            "reranker": {
                "enabled": False,
                "loaded": False,
                "model_id": "fake-reranker",
                "revision": None,
            },
        }


@pytest.fixture(autouse=True)
def fake_model_registry(monkeypatch):
    monkeypatch.setattr(app_main, "ModelRegistry", FakeRegistry)


def test_inference_limiter_bounds_active_and_queued_requests():
    async def scenario():
        limiter = InferenceLimiter(max_concurrency=1, queue_capacity=1)
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        async def first_request():
            async with limiter.slot():
                first_started.set()
                await release_first.wait()

        first = asyncio.create_task(first_request())
        await first_started.wait()

        second_slot = limiter.slot()
        second = asyncio.create_task(second_slot.__aenter__())
        await asyncio.sleep(0)
        with pytest.raises(InferenceQueueFull):
            async with limiter.slot():
                pass

        release_first.set()
        await first
        await second
        await second_slot.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_embedding_queue_returns_429_before_starting_more_inference(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("INFERENCE_QUEUE_CAPACITY", "0")
    started = threading.Event()
    release = threading.Event()

    class BlockingEmbeddingService:
        def embed(self, texts, input_type):
            started.set()
            assert release.wait(timeout=2)
            return type(
                "Result",
                (),
                {"model": "fake-embedding", "dimensions": 1, "embeddings": [[1.0]]},
            )()

    monkeypatch.setattr(
        embeddings_api, "_service", lambda request: BlockingEmbeddingService()
    )

    with TestClient(app_main.app) as client:
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(
                client.post, "/v1/embeddings/text", json={"text": "music"}
            )
            assert started.wait(timeout=2)
            overloaded = client.post("/v1/embeddings/text", json={"text": "music"})
            release.set()
            completed = first.result(timeout=2)

        assert completed.status_code == 200
        assert overloaded.status_code == 429
        assert overloaded.json() == {
            "detail": "Embedding inference queue is full. Retry later."
        }
        assert overloaded.headers["retry-after"] == "1"
        assert 'loopin_inference_rejected_total{operation="embeddings"} 1' in client.get(
            "/metrics"
        ).text


def test_embedding_and_reranker_limiters_are_independent():
    async def scenario():
        embeddings = InferenceLimiter(max_concurrency=1, queue_capacity=0)
        reranker = InferenceLimiter(max_concurrency=1, queue_capacity=0)
        async with embeddings.slot(), reranker.slot():
            with pytest.raises(InferenceQueueFull):
                async with embeddings.slot():
                    pass
            with pytest.raises(InferenceQueueFull):
                async with reranker.slot():
                    pass

    asyncio.run(scenario())


def test_runtime_settings_use_cpu_safe_defaults(monkeypatch):
    for variable in (
        "EMBEDDING_MAX_CONCURRENCY",
        "RERANKER_MAX_CONCURRENCY",
        "INFERENCE_QUEUE_CAPACITY",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
    ):
        monkeypatch.delenv(variable, raising=False)

    assert InferenceRuntimeSettings.from_environment() == InferenceRuntimeSettings(
        embedding_max_concurrency=2,
        reranker_max_concurrency=1,
        queue_capacity=20,
        omp_num_threads=2,
        mkl_num_threads=2,
        tokenizers_parallelism=False,
    )
