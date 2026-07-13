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
        ]
        with self._lock:
            timings = dict(self._timings)
            rejections = dict(self._rejections)
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
        return "\n".join(lines) + "\n"
