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

Configuration lives in `config/models.yaml`.

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