from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal

if TYPE_CHECKING:
    from app.models.registry import ModelRegistry


@dataclass(frozen=True)
class EmbeddingResult:
    model: str
    dimensions: int
    embeddings: list[list[float]]


class EmbeddingService:
    def __init__(self, registry: "ModelRegistry"):
        self.registry = registry

    def embed(
        self,
        texts: list[str],
        input_type: Literal["query", "passage"] = "passage",
    ) -> EmbeddingResult:
        config = self.registry.embedding_config
        prefix = "query: " if input_type == "query" else "passage: "
        normalized_texts = [f"{prefix}{text.strip()}" for text in texts]
        vectors = self.registry.embedding_model.encode(
            normalized_texts,
            normalize_embeddings=bool(config.get("normalize", True)),
        )

        return EmbeddingResult(
            model=config["model_id"],
            dimensions=int(config["dimensions"]),
            embeddings=vectors.tolist(),
        )
