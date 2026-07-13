from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.embeddings import router as embeddings_router
from app.api.rerank import router as rerank_router
from app.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model registry from config/models.yaml")
    registry = ModelRegistry.from_yaml("config/models.yaml")
    logger.info("Loading enabled models", extra={"models": registry.readiness()})
    registry.load_enabled()
    logger.info("Model loading complete", extra={"models": registry.readiness()})
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


@app.get("/ready", response_model=None)
def ready(request: Request) -> dict | JSONResponse:
    registry = request.app.state.models
    body = {
        "status": "ready" if registry.all_enabled_models_available() else "not_ready",
        **registry.readiness(),
    }
    if body["status"] == "ready":
        return body
    return JSONResponse(status_code=503, content=body)
