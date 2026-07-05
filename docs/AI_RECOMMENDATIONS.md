# Loopin AI Recommendations

Loopin AI v1 focuses on direct similarity actions:

- For You
- Similar Events
- Find Buddies
- Similar People

It intentionally does not include chatbot flows, LLM tool calling, or fine-tuning.

## Runtime model repos

Model artifacts are not committed to git. The service loads these Hugging Face model repos at runtime:

- Embedding model: [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
- Reranker model: [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)

## Copied bucket references

These bucket links are documentation/provenance only and are not used as runtime `model_id` values:

- Embedding bucket: [zxlnwy/multilingual-e5-small-bucket](https://huggingface.co/buckets/zxlnwy/multilingual-e5-small-bucket)
- Reranker bucket: [zxlnwy/bge-reranker-v2-m3-bucket](https://huggingface.co/buckets/zxlnwy/bge-reranker-v2-m3-bucket)

## Runtime flow

```text
loopin-api builds event or user text
loopin-ai generates embeddings
loopin-api stores vectors in pgvector
loopin-api retrieves top 50 eligible candidates
loopin-ai reranks candidates
loopin-api returns top 10 results to the frontend
```