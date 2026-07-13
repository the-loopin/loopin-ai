"""Small dependency-free Prometheus metrics exporter for inference runtime data."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from threading import Lock


@dataclass
class _Timing:
    count: int = 0
    queue_seconds: float = 0.0
    inference_seconds: float = 0.0


class InferenceMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._timings: dict[tuple[str, str], _Timing] = defaultdict(_Timing)
        self._rejections: dict[str, int] = defaultdict(int)
        self._active: dict[str, int] = defaultdict(int)
        self._model_loads: dict[str, tuple[bool, float]] = {}
        self._embedding_batch_sizes: dict[str, int] = defaultdict(int)
        self._reranker_candidate_counts: dict[str, int] = defaultdict(int)
        self._responses: dict[tuple[str, str, int], int] = defaultdict(int)

    def record_response(self, method: str, route: str, status_code: int) -> None:
        with self._lock:
            self._responses[(method, route, status_code)] += 1

    def change_active(self, operation: str, delta: int) -> None:
        with self._lock:
            self._active[operation] += delta

    def record_model_load(self, model: str, loaded: bool, duration_seconds: float) -> None:
        with self._lock:
            self._model_loads[model] = (loaded, duration_seconds)

    def record_embedding_batch_size(self, size: int) -> None:
        with self._lock:
            self._embedding_batch_sizes["embeddings"] += size

    def record_reranker_candidate_count(self, count: int) -> None:
        with self._lock:
            self._reranker_candidate_counts["reranker"] += count

    def record(
        self,
        operation: str,
        queue_seconds: float,
        inference_seconds: float,
        *,
        outcome: str,
    ) -> None:
        with self._lock:
            timing = self._timings[(operation, outcome)]
            timing.count += 1
            timing.queue_seconds += queue_seconds
            timing.inference_seconds += inference_seconds

    def record_rejection(self, operation: str) -> None:
        with self._lock:
            self._rejections[operation] += 1

    def render_prometheus(self) -> str:
        lines = [
            "# HELP loopin_inference_requests_total Completed inference requests.",
            "# TYPE loopin_inference_requests_total counter",
            "# HELP loopin_inference_queue_seconds Time requests spent waiting for inference.",
            "# TYPE loopin_inference_queue_seconds summary",
            "# HELP loopin_inference_duration_seconds Model inference execution time.",
            "# TYPE loopin_inference_duration_seconds summary",
            "# HELP loopin_inference_rejected_total Requests rejected because the inference queue was full.",
            "# TYPE loopin_inference_rejected_total counter",
            "# HELP loopin_inference_active Current active model executions.",
            "# TYPE loopin_inference_active gauge",
            "# HELP loopin_model_loaded Whether an enabled model loaded successfully.",
            "# TYPE loopin_model_loaded gauge",
            "# HELP loopin_model_load_duration_seconds Time spent loading a model.",
            "# TYPE loopin_model_load_duration_seconds gauge",
            "# HELP loopin_embedding_batch_size_total Number of texts submitted for embedding.",
            "# TYPE loopin_embedding_batch_size_total counter",
            "# HELP loopin_reranker_candidate_count_total Number of candidates submitted for reranking.",
            "# TYPE loopin_reranker_candidate_count_total counter",
            "# HELP loopin_http_responses_total HTTP responses by bounded route and status.",
            "# TYPE loopin_http_responses_total counter",
        ]
        with self._lock:
            timings = dict(self._timings)
            rejections = dict(self._rejections)
            active = dict(self._active)
            model_loads = dict(self._model_loads)
            batch_sizes = dict(self._embedding_batch_sizes)
            candidate_counts = dict(self._reranker_candidate_counts)
            responses = dict(self._responses)
        for (operation, outcome), timing in sorted(timings.items()):
            labels = f'operation="{operation}",outcome="{outcome}"'
            lines.extend(
                [
                    f"loopin_inference_requests_total{{{labels}}} {timing.count}",
                    f"loopin_inference_queue_seconds_sum{{{labels}}} {timing.queue_seconds}",
                    f"loopin_inference_queue_seconds_count{{{labels}}} {timing.count}",
                    f"loopin_inference_duration_seconds_sum{{{labels}}} {timing.inference_seconds}",
                    f"loopin_inference_duration_seconds_count{{{labels}}} {timing.count}",
                ]
            )
        for operation, count in sorted(rejections.items()):
            lines.append(
                f'loopin_inference_rejected_total{{operation="{operation}"}} {count}'
            )
        for operation, count in sorted(active.items()):
            lines.append(f'loopin_inference_active{{operation="{operation}"}} {count}')
        for model, (loaded, duration) in sorted(model_loads.items()):
            labels = f'model="{model}"'
            lines.append(f"loopin_model_loaded{{{labels}}} {int(loaded)}")
            lines.append(f"loopin_model_load_duration_seconds{{{labels}}} {duration}")
        for operation, count in sorted(batch_sizes.items()):
            lines.append(f'loopin_embedding_batch_size_total{{operation="{operation}"}} {count}')
        for operation, count in sorted(candidate_counts.items()):
            lines.append(f'loopin_reranker_candidate_count_total{{operation="{operation}"}} {count}')
        for (method, route, status), count in sorted(responses.items()):
            lines.append(
                f'loopin_http_responses_total{{method="{method}",route="{route}",status="{status}"}} {count}'
            )
        return "\n".join(lines) + "\n"
