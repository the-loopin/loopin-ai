# Loopin AI

FastAPI microservice for Loopin semantic recommendations. It owns model inference only:

- text embeddings
- batch embeddings
- reranking retrieved candidates

It does not own users, events, permissions, pgvector retrieval, chat, LLM tool calling, or fine-tuning.

## Models

Model binaries are not committed to this repository. The service loads the configured Hugging Face model repos at runtime:

- Embedding model: [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
- Reranker model: [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)

The copied bucket links are kept in docs as provenance only. Runtime loading uses model repos, not bucket ids.

Configuration lives in `config/models.yaml`. Embeddings are enabled by default and the
reranker is disabled by default, so it consumes no model memory in the current deployment.
Each model section has an `enabled` flag and an `active` model name:

```yaml
embeddings:
  enabled: true
  active: multilingual_e5_small

reranker:
  enabled: false
  active: bge_reranker_v2_m3
```

`/ready` reports each model's `enabled`, `loaded`, `model_id`, `revision`, and embedding
`dimensions`.

To enable reranking in a later deployment without a code change, either set
`reranker.enabled: true` in the configuration or supply environment overrides:

```bash
LOOPIN_RERANKER_ENABLED=true
# Optional model selection overrides:
LOOPIN_EMBEDDINGS_ACTIVE=multilingual_e5_small
LOOPIN_RERANKER_ACTIVE=bge_reranker_v2_m3
```

When reranking is disabled or its model cannot load, `POST /v1/rerank` returns `503 Service
Unavailable`. An unavailable embedding model similarly returns 503 from embedding endpoints.

## CPU inference runtime

Model inference is deliberately bounded independently for embeddings and reranking. A request
that cannot enter its operation's bounded queue returns `429 Too Many Requests` with
`Retry-After: 1`, rather than waiting indefinitely. Each service process has its own queues, so
run exactly one Uvicorn worker for a CPU deployment: more workers each load separate copies of
the configured models and multiply CPU pressure and memory use.

Recommended settings for a 2-vCPU server (these are the Docker Compose defaults):

```bash
# Uvicorn workers: 1 (the Docker image enforces this)
EMBEDDING_MAX_CONCURRENCY=2
RERANKER_MAX_CONCURRENCY=1
INFERENCE_QUEUE_CAPACITY=20
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
TOKENIZERS_PARALLELISM=false
```

`INFERENCE_QUEUE_CAPACITY` applies independently to each inference type and excludes requests
already running. `EMBEDDING_MAX_CONCURRENCY` and `RERANKER_MAX_CONCURRENCY` control active model
calls, with the reranker defaulting to one. CPU thread environment values are applied before the
model libraries load.

`GET /metrics` exports Prometheus-compatible counters and summaries. The
`loopin_inference_queue_seconds_*` series measures admission wait time, while
`loopin_inference_duration_seconds_*` measures only model execution. Rejections are exposed by
`loopin_inference_rejected_total`.

### CPU load test

Start a local deployment with reranking enabled when testing both operations, then run:

```bash
python tests/load_inference.py --operation both --requests 40 --concurrency 8
```

The script prints HTTP status counts plus `/metrics`. With the 2-vCPU defaults, the expected
result under sustained overload is a mix of successful responses and explicit `429` responses;
there should be no unbounded pending work or request timeouts. Capture the printed JSON and the
queue/inference timing series as the load-test result for the target hardware, since model cache,
CPU generation, and candidate sizes determine the exact throughput.

### Recorded CPU load-test result (2026-07-13)

The provided load runner was executed against the real `BAAI/bge-reranker-v2-m3` model in the
Docker image (Python 3.12.13) on a Docker-limited 2-vCPU `x86_64` Linux container. The host was a
13th Gen Intel Core i7-13700 (16 cores / 24 logical processors). The deployment used one Uvicorn
worker, `RERANKER_MAX_CONCURRENCY=1`, `INFERENCE_QUEUE_CAPACITY=2`,
`OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, and `TOKENIZERS_PARALLELISM=false`.

`python tests/load_inference.py --operation reranker --requests 40 --concurrency 8` produced:

| Measure | Result |
| --- | ---: |
| Successful responses | 3 (`200`) |
| Controlled overload responses | 37 (`429`) |
| End-to-end elapsed time | 0.954 s |
| Reranker queue time | 1.172830 s total / 3 requests (0.390943 s average) |
| Reranker inference time | 0.775260 s total / 3 requests (0.258420 s average) |
| Reranker queue rejections | 37 |

All 40 requests completed within the 0.954-second run; none waited indefinitely or timed out.
The small queue intentionally rejected excess concurrent work, demonstrating the overload path.

## Endpoints

## Internal deployment and authentication

This is an internal-only service. Deploy it on a private network (or behind an authenticated
service mesh/reverse proxy) and do not expose port 8000 directly to the public internet. The
Compose file intentionally requires `LOOPIN_SERVICE_TOKEN`, which should come from the runtime
secret manager or an uncommitted `.env` file:

```bash
LOOPIN_SERVICE_TOKEN='replace-with-a-long-random-secret'
LOOPIN_ENV=production
```

Every inference request and `/metrics` must send this credential using either
`Authorization: Bearer <token>` or `X-Loopin-Service-Token: <token>`. For example, an API client
reads its credential from its environment rather than source code:

```python
import os
import httpx

headers = {"Authorization": f"Bearer {os.environ['LOOPIN_SERVICE_TOKEN']}"}
response = httpx.post("http://loopin-ai.internal/v1/embeddings/text", headers=headers,
                      json={"text": "recommendation text", "input_type": "passage"})
response.raise_for_status()
```

`/health` is deliberately process-only and remains `200` while a model dependency is unavailable.
`/ready` returns `503` unless every enabled model has loaded. This makes `/health` appropriate for
process liveness and `/ready` appropriate for traffic admission. In production (`LOOPIN_ENV=production`),
OpenAPI, Swagger UI, and ReDoc are disabled; keep development documentation available only in a
trusted local environment.

Each response includes `X-Request-ID`. Supply that header from the caller to propagate an existing
correlation ID; otherwise the service creates one. Logs record only request metadata, IDs, counts,
durations, and error types—never submitted recommendation, query, or candidate text.

`/metrics` is authenticated and emits Prometheus metrics for HTTP response status, inference
request/outcome, queue and execution latency, active inference, overload rejection, model load
status/duration, embedding batch volume, and reranker candidate volume. Configure Prometheus to
scrape it over the same private network with the service token.

The provided Compose service uses `expose` instead of a host `ports` mapping. Attach the calling
service to the Compose network; use an explicit, access-controlled local override only when
debugging from a workstation.

### `POST /v1/embeddings/text`

```json
{
  "text": "Rooftop jazz night in Baku",
  "input_type": "passage"
}
```

### `POST /v1/embeddings/batch`

```json
{
  "texts": ["Jazz night", "Startup meetup"],
  "input_type": "passage"
}
```

### `POST /v1/rerank`

```json
{
  "query": "I want live music and relaxed social events",
  "top_k": 10,
  "candidates": [
    {
      "id": "event_123",
      "text": "Rooftop jazz night with small groups",
      "metadata": {
        "retrieval_score": 0.82
      }
    }
  ]
}
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Or:

```bash
docker compose up --build
```

## Tests and real-model benchmarks

The default suite uses fake models and remains fast; it never asks Hugging Face for model
artifacts:

```bash
pytest -q
```

The real embedding contracts are explicitly opt-in. They use the active embedding model in
`config/models.yaml`, exercise both the service and authenticated HTTP endpoints, and may download
the model into the standard Hugging Face cache on their first run:

```bash
RUN_REAL_MODEL_TESTS=true pytest -q -m real_model
```

The reranker remains separately opt-in and production configuration remains disabled by default:

```bash
RUN_REAL_MODEL_TESTS=true \
RUN_RERANKER_TESTS=true \
pytest -q -m "real_model or reranker"
```

Run the fixture-based vector benchmark with the configured embedding model:

```bash
python benchmarks/run_recommendation_benchmark.py
```

Run its optional reranker comparison:

```bash
RUN_RERANKER_TESTS=true \
python benchmarks/run_recommendation_benchmark.py --include-reranker
```

The runner writes a timestamped JSON record under `benchmark-results/` and refreshes
`docs/benchmarks/latest.md`. Outputs include only IDs and aggregates, never raw query, event, or
candidate text. `Recall@10` is the fraction of every query's relevant candidates retrieved among
the first ten; MRR is the mean reciprocal rank of the first relevant result. It records
warm-up-excluded CPU latency and a background-sampled process RSS peak (MiB), model
identity/revision details, and a thresholded reranker recommendation. Expect several hundred MiB
of RAM for the embedding model and substantially more when the reranker is enabled; exact
requirements and CPU latency depend on the host. Heavy tests are excluded from the default suite
to prevent model downloads and routine developer/CI delays.

## Runtime flow

```text
loopin-api builds recommendation query text
loopin-api retrieves top 50 candidates from pgvector
loopin-api sends query and candidate texts to loopin-ai /v1/rerank
loopin-ai returns ranked candidates
loopin-api returns top 10 results to frontend
```
