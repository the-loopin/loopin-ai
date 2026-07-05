from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request

from app.api.embeddings import router as embeddings_router
from app.api.rerank import router as rerank_router
from app.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model registry from config/models.yaml")
    registry = ModelRegistry.from_yaml("config/models.yaml")
    logger.info("Loading embedding and reranker models", extra=registry.readiness())
    registry.load_all()
    logger.info("Models loaded", extra=registry.readiness())
    app.state.models = registry
    yield


app = FastAPI(
    title="Loopin AI",
    description="Embedding and reranking microservice for Loopin recommendations.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(embeddings_router, prefix="/v1/embeddings", tags=["embeddings"])
app.include_router(rerank_router, prefix="/v1", tags=["rerank"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> dict[str, str]:
    return {"status": "ready", **request.app.state.models.readiness()}
