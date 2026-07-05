from functools import cached_property
from pathlib import Path
from typing import Any

import yaml


class ModelConfigError(ValueError):
    pass


class ModelRegistry:
    def __init__(self, config: dict[str, Any]):
        self._validate(config)
        self.config = config

    @classmethod
    def from_yaml(cls, path: str) -> "ModelRegistry":
        with Path(path).open("r", encoding="utf-8") as file:
            return cls(yaml.safe_load(file))

    @classmethod
    def _validate(cls, config: dict[str, Any]) -> None:
        if not isinstance(config, dict):
            raise ModelConfigError("Model config must be a YAML mapping.")

        embedding_config = cls._active_model_config(config, "embeddings")
        reranker_config = cls._active_model_config(config, "reranker")

        cls._require_non_empty_string(embedding_config, "model_id", "embeddings")
        cls._require_positive_int(embedding_config, "dimensions", "embeddings")

        cls._require_non_empty_string(reranker_config, "model_id", "reranker")
        cls._require_positive_int(reranker_config, "return_top_k", "reranker")

        if "rerank_top_k" in reranker_config:
            cls._require_positive_int(reranker_config, "rerank_top_k", "reranker")

    @staticmethod
    def _active_model_config(
        config: dict[str, Any], section_name: str
    ) -> dict[str, Any]:
        section = config.get(section_name)
        if not isinstance(section, dict):
            raise ModelConfigError(f"Missing or invalid '{section_name}' config section.")

        active = section.get("active")
        models = section.get("models")
        if not isinstance(active, str) or not active:
            raise ModelConfigError(f"Missing active model name for '{section_name}'.")
        if not isinstance(models, dict):
            raise ModelConfigError(f"Missing models mapping for '{section_name}'.")

        model_config = models.get(active)
        if not isinstance(model_config, dict):
            raise ModelConfigError(
                f"Active model '{active}' is not defined in '{section_name}'."
            )
        return model_config

    @staticmethod
    def _require_non_empty_string(
        config: dict[str, Any], field_name: str, section_name: str
    ) -> None:
        value = config.get(field_name)
        if not isinstance(value, str) or not value:
            raise ModelConfigError(
                f"Missing or invalid '{field_name}' for '{section_name}' model."
            )

    @staticmethod
    def _require_positive_int(
        config: dict[str, Any], field_name: str, section_name: str
    ) -> None:
        value = config.get(field_name)
        if type(value) is not int or value < 1:
            raise ModelConfigError(
                f"Missing or invalid '{field_name}' for '{section_name}' model."
            )

    @property
    def embedding_config(self) -> dict[str, Any]:
        embeddings = self.config["embeddings"]
        return embeddings["models"][embeddings["active"]]

    @property
    def reranker_config(self) -> dict[str, Any]:
        reranker = self.config["reranker"]
        return reranker["models"][reranker["active"]]

    @cached_property
    def embedding_model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.embedding_config["model_id"])

    @cached_property
    def reranker_model(self):
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.reranker_config["model_id"])

    def load_all(self) -> None:
        _ = self.embedding_model
        _ = self.reranker_model

    def readiness(self) -> dict[str, str]:
        return {
            "embedding_model": self.embedding_config["model_id"],
            "reranker_model": self.reranker_config["model_id"],
        }
