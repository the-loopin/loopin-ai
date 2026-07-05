from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.registry import ModelRegistry



@dataclass(frozen=True)
class RerankResult:
    model: str
    results: list[dict]


class RerankerService:
    def __init__(self, registry: "ModelRegistry"):
        self.registry = registry

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> RerankResult:
        config = self.registry.reranker_config
        limit = min(top_k or int(config.get("return_top_k", 10)), len(candidates))
        pairs = [(query, candidate["text"]) for candidate in candidates]
        scores = self.registry.reranker_model.predict(pairs)

        ranked = sorted(
            zip(candidates, [float(score) for score in scores], strict=True),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]

        return RerankResult(
            model=config["model_id"],
            results=[
                {
                    "id": candidate["id"],
                    "score": score,
                    "rank": index + 1,
                    "metadata": candidate.get("metadata"),
                }
                for index, (candidate, score) in enumerate(ranked)
            ],
        )
