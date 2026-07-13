from contextlib import asynccontextmanager
import logging
import os
from time import perf_counter

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from app.api.embeddings import router as embeddings_router
from app.api.rerank import router as rerank_router
from app.models.registry import ModelRegistry
from app.metrics import InferenceMetrics
from app.runtime import InferenceLimiter, InferenceRuntimeSettings
from app.security import (
    SecuritySettings,
    inbound_request_id,
    require_service_token,
    reset_request_id,
    set_request_id,
)

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
    app.state.security = SecuritySettings.from_environment()
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
    for model, state in registry.readiness().items():
        if state["enabled"]:
            app.state.inference_metrics.record_model_load(
                model,
                state["loaded"],
                getattr(registry, "load_duration", lambda _: 0.0)(model),
            )
    logger.info("Model loading complete", extra={"models": registry.readiness()})
    app.state.models = registry
    yield


_production = os.getenv("LOOPIN_ENV", "development").strip().lower() == "production"
app = FastAPI(
    title="Loopin AI",
    description="Embedding and reranking microservice for Loopin recommendations.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None if _production else "/docs",
    redoc_url=None if _production else "/redoc",
    openapi_url=None if _production else "/openapi.json",
)

app.include_router(embeddings_router, prefix="/v1/embeddings", tags=["embeddings"])
app.include_router(rerank_router, prefix="/v1", tags=["rerank"])


@app.middleware("http")
async def request_correlation(request: Request, call_next):
    correlation_id = inbound_request_id(request.headers.get("x-request-id"))
    token = set_request_id(correlation_id)
    started_at = perf_counter()
    try:
        response = await call_next(request)
        if request.url.path != "/metrics":
            request.app.state.inference_metrics.record_response(
                request.method, request.url.path, response.status_code
            )
        logger.info(
            "HTTP request completed",
            extra={
                "request_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_seconds": perf_counter() - started_at,
            },
        )
    finally:
        reset_request_id(token)
    response.headers["X-Request-ID"] = correlation_id
    return response


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


@app.get("/metrics", include_in_schema=False, dependencies=[Depends(require_service_token)])
def metrics(request: Request) -> Response:
    return Response(
        content=request.app.state.inference_metrics.render_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
