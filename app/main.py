from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from app.api.embeddings import router as embeddings_router
from app.api.rerank import router as rerank_router
from app.models.registry import ModelRegistry
from app.metrics import InferenceMetrics
from app.runtime import InferenceLimiter, InferenceRuntimeSettings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = InferenceRuntimeSettings.from_environment()
    settings.apply_cpu_environment()
    app.state.inference_limiters = {
        "embeddings": InferenceLimiter(
            settings.embedding_max_concurrency, settings.queue_capacity
        ),
        "reranker": InferenceLimiter(
            settings.reranker_max_concurrency, settings.queue_capacity
        ),
    }
    app.state.inference_metrics = InferenceMetrics()
    logger.info(
        "Configured bounded CPU inference runtime",
        extra={
            "embedding_max_concurrency": settings.embedding_max_concurrency,
            "reranker_max_concurrency": settings.reranker_max_concurrency,
            "inference_queue_capacity": settings.queue_capacity,
            "omp_num_threads": settings.omp_num_threads,
            "mkl_num_threads": settings.mkl_num_threads,
            "tokenizers_parallelism": settings.tokenizers_parallelism,
        },
    )
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
        "status": registry.readiness_status(),
        **registry.readiness(),
    }
    if body["status"] != "not_ready":
        return body
    return JSONResponse(status_code=503, content=body)


@app.get("/metrics", include_in_schema=False)
def metrics(request: Request) -> Response:
    return Response(
        content=request.app.state.inference_metrics.render_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
