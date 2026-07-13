import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.inference import run_bounded_inference
from app.runtime import InferenceQueueFull
from app.services.embedding_service import EmbeddingService
from app.security import request_id
from app.security import require_service_token

router = APIRouter(dependencies=[Depends(require_service_token)])
logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 4096
MAX_BATCH_SIZE = 128
EmbeddingText = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_LENGTH)]


class TextEmbeddingRequest(BaseModel):
    text: EmbeddingText
    input_type: Literal["query", "passage"] = "passage"


class BatchEmbeddingRequest(BaseModel):
    texts: list[EmbeddingText] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)
    input_type: Literal["query", "passage"] = "passage"


class EmbeddingResponse(BaseModel):
    model: str
    dimensions: int
    embedding: list[float]


class BatchEmbeddingResponse(BaseModel):
    model: str
    dimensions: int
    embeddings: list[list[float]]


def _service(request: Request) -> EmbeddingService:
    return EmbeddingService(request.app.state.models)


@router.post("/text", response_model=EmbeddingResponse)
async def embed_text(payload: TextEmbeddingRequest, request: Request) -> EmbeddingResponse:
    registry = request.app.state.models
    if not registry.is_available("embeddings"):
        raise HTTPException(status_code=503, detail=registry.unavailable_reason("embeddings"))
    logger.info(
        "Embedding text request",
        extra={"request_id": request_id(), "input_type": payload.input_type, "items": 1},
    )
    try:
        request.app.state.inference_metrics.record_embedding_batch_size(1)
        result = await run_bounded_inference(
            request.app,
            "embeddings",
            lambda: _service(request).embed([payload.text], input_type=payload.input_type),
        )
    except InferenceQueueFull:
        raise HTTPException(
            status_code=429,
            detail="Embedding inference queue is full. Retry later.",
            headers={"Retry-After": "1"},
        )
    except Exception as exc:
        logger.error("Embedding text request failed", extra={"request_id": request_id(), "error_type": exc.__class__.__name__})
        raise HTTPException(status_code=500, detail="Embedding inference failed.") from exc
    return EmbeddingResponse(
        model=result.model,
        dimensions=result.dimensions,
        embedding=result.embeddings[0],
    )


@router.post("/batch", response_model=BatchEmbeddingResponse)
async def embed_batch(payload: BatchEmbeddingRequest, request: Request) -> BatchEmbeddingResponse:
    registry = request.app.state.models
    if not registry.is_available("embeddings"):
        raise HTTPException(status_code=503, detail=registry.unavailable_reason("embeddings"))
    logger.info(
        "Embedding batch request",
        extra={"request_id": request_id(), "input_type": payload.input_type, "items": len(payload.texts)},
    )
    try:
        request.app.state.inference_metrics.record_embedding_batch_size(len(payload.texts))
        result = await run_bounded_inference(
            request.app,
            "embeddings",
            lambda: _service(request).embed(payload.texts, input_type=payload.input_type),
        )
    except InferenceQueueFull:
        raise HTTPException(
            status_code=429,
            detail="Embedding inference queue is full. Retry later.",
            headers={"Retry-After": "1"},
        )
    except Exception as exc:
        logger.error("Embedding batch request failed", extra={"request_id": request_id(), "error_type": exc.__class__.__name__})
        raise HTTPException(status_code=500, detail="Embedding inference failed.") from exc
    return BatchEmbeddingResponse(
        model=result.model,
        dimensions=result.dimensions,
        embeddings=result.embeddings,
    )
