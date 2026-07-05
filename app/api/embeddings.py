from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.services.embedding_service import EmbeddingService

router = APIRouter()


class TextEmbeddingRequest(BaseModel):
    text: str = Field(..., min_length=1)
    input_type: Literal["query", "passage"] = "passage"


class BatchEmbeddingRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
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
    result = _service(request).embed([payload.text], input_type=payload.input_type)
    return EmbeddingResponse(
        model=result.model,
        dimensions=result.dimensions,
        embedding=result.embeddings[0],
    )


@router.post("/batch", response_model=BatchEmbeddingResponse)
def embed_batch(payload: BatchEmbeddingRequest, request: Request) -> BatchEmbeddingResponse:
    result = _service(request).embed(payload.texts, input_type=payload.input_type)
    return BatchEmbeddingResponse(
        model=result.model,
        dimensions=result.dimensions,
        embeddings=result.embeddings,
    )
