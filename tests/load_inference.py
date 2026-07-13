"""Basic live-service CPU load test for embeddings and reranking.

Start the service with the desired model configuration first, then run this file directly.
It deliberately sends more concurrent requests than the configured inference limits so that
queueing and controlled 429 responses can be observed.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

import httpx


def _request(client: httpx.Client, base_url: str, operation: str) -> int:
    if operation == "embeddings":
        response = client.post(
            f"{base_url}/v1/embeddings/text",
            json={"text": "Rooftop jazz night with friendly small groups"},
        )
    else:
        response = client.post(
            f"{base_url}/v1/rerank",
            json={
                "query": "live jazz events",
                "candidates": [
                    {"id": "event_1", "text": "Rooftop jazz night"},
                    {"id": "event_2", "text": "Morning yoga class"},
                ],
            },
        )
    return response.status_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--operation", choices=("embeddings", "reranker", "both"), default="both"
    )
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    service_token = os.environ.get("LOOPIN_SERVICE_TOKEN")
    if not service_token:
        parser.error("LOOPIN_SERVICE_TOKEN must be set for authenticated requests.")

    operations = (
        ["embeddings", "reranker"] * ((args.requests + 1) // 2)
        if args.operation == "both"
        else [args.operation] * args.requests
    )[: args.requests]
    started_at = perf_counter()
    with httpx.Client(
        timeout=30.0,
        headers={"Authorization": f"Bearer {service_token}"},
    ) as client, ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(_request, client, args.base_url.rstrip("/"), operation)
            for operation in operations
        ]
        statuses = Counter(future.result() for future in as_completed(futures))
        metrics = client.get(f"{args.base_url.rstrip('/')}/metrics").text

    print(
        json.dumps(
            {
                "requests": args.requests,
                "concurrency": args.concurrency,
                "elapsed_seconds": round(perf_counter() - started_at, 3),
                "status_counts": dict(sorted(statuses.items())),
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
