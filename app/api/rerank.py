import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.inference import run_bounded_inference
from app.runtime import InferenceQueueFull
from app.services.reranker_service import RerankerService
from app.security import request_id
from app.security import require_service_token

router = APIRouter(dependencies=[Depends(require_service_token)])
logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 4096
MAX_CANDIDATES = 100
MAX_TOP_K = 50


class RerankCandidate(BaseModel):
    id: str
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    metadata: dict | None = None


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    candidates: list[RerankCandidate] = Field(
        ..., min_length=1, max_length=MAX_CANDIDATES
    )
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)


class RankedCandidate(BaseModel):
    id: str
    score: float
    rank: int
    metadata: dict | None = None


class RerankResponse(BaseModel):
    model: str
    results: list[RankedCandidate]


def _service(request: Request) -> RerankerService:
    return RerankerService(request.app.state.models)


@router.post("/rerank", response_model=RerankResponse)
async def rerank(payload: RerankRequest, request: Request) -> RerankResponse:
    registry = request.app.state.models
    if not registry.is_available("reranker"):
        raise HTTPException(status_code=503, detail=registry.unavailable_reason("reranker"))
    logger.info(
        "Rerank request",
        extra={"request_id": request_id(), "candidates": len(payload.candidates), "top_k": payload.top_k},
    )
    try:
        request.app.state.inference_metrics.record_reranker_candidate_count(len(payload.candidates))
        result = await run_bounded_inference(
            request.app,
            "reranker",
            lambda: _service(request).rerank(
                query=payload.query,
                candidates=[candidate.model_dump() for candidate in payload.candidates],
                top_k=payload.top_k,
            ),
        )
    except InferenceQueueFull:
        raise HTTPException(
            status_code=429,
            detail="Reranker inference queue is full. Retry later.",
            headers={"Retry-After": "1"},
        )
    except Exception as exc:
        logger.error("Rerank request failed", extra={"request_id": request_id(), "error_type": exc.__class__.__name__})
        raise HTTPException(status_code=500, detail="Rerank inference failed.") from exc
    return RerankResponse(model=result.model, results=result.results)
