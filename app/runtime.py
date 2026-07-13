"""CPU inference runtime configuration and bounded admission control."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import AsyncIterator


class RuntimeConfigError(ValueError):
    """Raised when an inference runtime environment variable is invalid."""


class InferenceQueueFull(Exception):
    """Raised before a request can enter a full inference queue."""


def _positive_int_from_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeConfigError(f"{name} must be an integer.") from exc
    if value < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise RuntimeConfigError(f"{name} must be {qualifier}.")
    return value


@dataclass(frozen=True)
class InferenceRuntimeSettings:
    embedding_max_concurrency: int
    reranker_max_concurrency: int
    queue_capacity: int
    omp_num_threads: int
    mkl_num_threads: int
    tokenizers_parallelism: bool

    @classmethod
    def from_environment(cls) -> "InferenceRuntimeSettings":
        tokenizers_parallelism = os.getenv("TOKENIZERS_PARALLELISM", "false").lower()
        if tokenizers_parallelism not in {"true", "false"}:
            raise RuntimeConfigError(
                "TOKENIZERS_PARALLELISM must be 'true' or 'false'."
            )
        return cls(
            embedding_max_concurrency=_positive_int_from_env(
                "EMBEDDING_MAX_CONCURRENCY", 2
            ),
            reranker_max_concurrency=_positive_int_from_env(
                "RERANKER_MAX_CONCURRENCY", 1
            ),
            queue_capacity=_positive_int_from_env(
                "INFERENCE_QUEUE_CAPACITY", 20, minimum=0
            ),
            omp_num_threads=_positive_int_from_env("OMP_NUM_THREADS", 2),
            mkl_num_threads=_positive_int_from_env("MKL_NUM_THREADS", 2),
            tokenizers_parallelism=tokenizers_parallelism == "true",
        )

    def apply_cpu_environment(self) -> None:
        """Set thread controls before sentence-transformers imports its ML runtime."""
        os.environ["OMP_NUM_THREADS"] = str(self.omp_num_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.mkl_num_threads)
        os.environ["TOKENIZERS_PARALLELISM"] = str(self.tokenizers_parallelism).lower()


class InferenceLimiter:
    """Limits active inferences and the number of requests allowed to wait."""

    def __init__(self, max_concurrency: int, queue_capacity: int):
        self.max_concurrency = max_concurrency
        self.queue_capacity = queue_capacity
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._lock = asyncio.Lock()
        self._active = 0
        self._queued = 0

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[float]:
        """Reserve a bounded queue position and yield the time spent waiting."""
        queued_at = perf_counter()
        async with self._lock:
            if (
                self._active + self._queued
                >= self.max_concurrency + self.queue_capacity
            ):
                raise InferenceQueueFull
            self._queued += 1

        acquired = False
        active = False
        try:
            await self._semaphore.acquire()
            acquired = True
            queue_seconds = perf_counter() - queued_at
            async with self._lock:
                self._queued -= 1
                self._active += 1
                active = True
            yield queue_seconds
        finally:
            async with self._lock:
                if active:
                    self._active -= 1
                else:
                    self._queued -= 1
            if acquired:
                self._semaphore.release()
