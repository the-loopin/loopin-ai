# syntax=docker/dockerfile:1.7

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/opt/loopin/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY config ./config

RUN python - <<'PY'
from pathlib import Path

import yaml
from sentence_transformers import CrossEncoder, SentenceTransformer


config_path = Path("/app/config/models.yaml")

with config_path.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


def model_kwargs(model_config: dict) -> dict:
    revision = model_config.get("revision")
    return {"revision": revision} if revision else {}


embeddings = config["embeddings"]

if embeddings["enabled"]:
    active_model = embeddings["models"][embeddings["active"]]
    model_id = active_model["model_id"]

    print(f"Baking embedding model into image: {model_id}")
    SentenceTransformer(model_id, **model_kwargs(active_model))


reranker = config["reranker"]

if reranker["enabled"]:
    active_model = reranker["models"][reranker["active"]]
    model_id = active_model["model_id"]

    print(f"Baking reranker model into image: {model_id}")
    CrossEncoder(model_id, **model_kwargs(active_model))


print("Enabled models successfully baked into Docker image.")
PY

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY app ./app

EXPOSE 8000

CMD [
    "uvicorn",
    "app.main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
    "--workers",
    "1"
]