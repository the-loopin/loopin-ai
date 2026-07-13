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

## Endpoints

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

## Runtime flow

```text
loopin-api builds recommendation query text
loopin-api retrieves top 50 candidates from pgvector
loopin-api sends query and candidate texts to loopin-ai /v1/rerank
loopin-ai returns ranked candidates
loopin-api returns top 10 results to frontend
```
