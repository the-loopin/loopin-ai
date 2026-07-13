"""Inference execution, overload handling, and timing instrumentation."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Callable, TypeVar

from starlette.concurrency import run_in_threadpool

from app.runtime import InferenceQueueFull

logger = logging.getLogger(__name__)
Result = TypeVar("Result")


async def run_bounded_inference(app, operation: str, call: Callable[[], Result]) -> Result:
    """Run a blocking model call only after it has been admitted to its queue."""
    limiter = app.state.inference_limiters[operation]
    try:
        async with limiter.slot() as queue_seconds:
            started_at = perf_counter()
            try:
                result = await run_in_threadpool(call)
            except Exception:
                inference_seconds = perf_counter() - started_at
                app.state.inference_metrics.record(
                    operation, queue_seconds, inference_seconds, outcome="error"
                )
                logger.exception(
                    "Inference request failed",
                    extra={
                        "operation": operation,
                        "queue_seconds": queue_seconds,
                        "inference_seconds": inference_seconds,
                    },
                )
                raise
            inference_seconds = perf_counter() - started_at
            app.state.inference_metrics.record(
                operation, queue_seconds, inference_seconds, outcome="success"
            )
            logger.info(
                "Inference request completed",
                extra={
                    "operation": operation,
                    "queue_seconds": queue_seconds,
                    "inference_seconds": inference_seconds,
                },
            )
            return result
    except InferenceQueueFull:
        app.state.inference_metrics.record_rejection(operation)
        logger.warning("Inference queue full", extra={"operation": operation})
        raise
