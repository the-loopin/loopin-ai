import logging
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.services.embedding_service import EmbeddingService

router = APIRouter()
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
def embed_text(payload: TextEmbeddingRequest, request: Request) -> EmbeddingResponse:
    registry = request.app.state.models
    if not registry.is_available("embeddings"):
        raise HTTPException(status_code=503, detail=registry.unavailable_reason("embeddings"))
    logger.info(
        "Embedding text request",
        extra={"input_type": payload.input_type, "items": 1},
    )
    try:
        result = _service(request).embed([payload.text], input_type=payload.input_type)
    except Exception as exc:
        logger.exception("Embedding text request failed")
        raise HTTPException(status_code=500, detail="Embedding inference failed.") from exc
    return EmbeddingResponse(
        model=result.model,
        dimensions=result.dimensions,
        embedding=result.embeddings[0],
    )


@router.post("/batch", response_model=BatchEmbeddingResponse)
def embed_batch(payload: BatchEmbeddingRequest, request: Request) -> BatchEmbeddingResponse:
    registry = request.app.state.models
    if not registry.is_available("embeddings"):
        raise HTTPException(status_code=503, detail=registry.unavailable_reason("embeddings"))
    logger.info(
        "Embedding batch request",
        extra={"input_type": payload.input_type, "items": len(payload.texts)},
    )
    try:
        result = _service(request).embed(payload.texts, input_type=payload.input_type)
    except Exception as exc:
        logger.exception("Embedding batch request failed")
        raise HTTPException(status_code=500, detail="Embedding inference failed.") from exc
    return BatchEmbeddingResponse(
        model=result.model,
        dimensions=result.dimensions,
        embeddings=result.embeddings,
    )
