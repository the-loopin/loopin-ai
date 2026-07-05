from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.services.reranker_service import RerankerService

router = APIRouter()


class RerankCandidate(BaseModel):
    id: str
    text: str = Field(..., min_length=1)
    metadata: dict | None = None


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1)
    candidates: list[RerankCandidate] = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1)


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
def rerank(payload: RerankRequest, request: Request) -> RerankResponse:
    result = _service(request).rerank(
        query=payload.query,
        candidates=[candidate.model_dump() for candidate in payload.candidates],
        top_k=payload.top_k,
    )
    return RerankResponse(model=result.model, results=result.results)
