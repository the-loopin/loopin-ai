import copy
import os
from pathlib import Path
from typing import Any

import yaml


class ModelConfigError(ValueError):
    pass


class ModelRegistry:
    """Owns configured models and their independent runtime availability."""

    def __init__(self, config: dict[str, Any]):
        self.config = copy.deepcopy(config)
        self._validate(self.config)
        self._models: dict[str, Any] = {}
        self._load_errors: dict[str, str] = {}

    @classmethod
    def from_yaml(cls, path: str) -> "ModelRegistry":
        with Path(path).open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return cls(cls._with_environment_overrides(config))

    @staticmethod
    def _with_environment_overrides(config: Any) -> Any:
        config = copy.deepcopy(config)
        if not isinstance(config, dict):
            return config

        for section_name, env_prefix in (
            ("embeddings", "LOOPIN_EMBEDDINGS"),
            ("reranker", "LOOPIN_RERANKER"),
        ):
            section = config.get(section_name)
            if not isinstance(section, dict):
                continue
            enabled = os.getenv(f"{env_prefix}_ENABLED")
            if enabled is not None:
                normalized = enabled.strip().lower()
                if normalized not in {"true", "false"}:
                    raise ModelConfigError(
                        f"{env_prefix}_ENABLED must be 'true' or 'false'."
                    )
                section["enabled"] = normalized == "true"
            active = os.getenv(f"{env_prefix}_ACTIVE")
            if active is not None:
                section["active"] = active
        return config

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
    def _active_model_config(config: dict[str, Any], section_name: str) -> dict[str, Any]:
        section = config.get(section_name)
        if not isinstance(section, dict):
            raise ModelConfigError(f"Missing or invalid '{section_name}' config section.")
        if type(section.get("enabled")) is not bool:
            raise ModelConfigError(f"Missing or invalid 'enabled' for '{section_name}'.")

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
    def _require_non_empty_string(config: dict[str, Any], field_name: str, section_name: str) -> None:
        value = config.get(field_name)
        if not isinstance(value, str) or not value:
            raise ModelConfigError(
                f"Missing or invalid '{field_name}' for '{section_name}' model."
            )

    @staticmethod
    def _require_positive_int(config: dict[str, Any], field_name: str, section_name: str) -> None:
        value = config.get(field_name)
        if type(value) is not int or value < 1:
            raise ModelConfigError(
                f"Missing or invalid '{field_name}' for '{section_name}' model."
            )

    def _section(self, name: str) -> dict[str, Any]:
        return self.config[name]

    def enabled(self, name: str) -> bool:
        return bool(self._section(name)["enabled"])

    @property
    def embedding_config(self) -> dict[str, Any]:
        section = self._section("embeddings")
        return section["models"][section["active"]]

    @property
    def reranker_config(self) -> dict[str, Any]:
        section = self._section("reranker")
        return section["models"][section["active"]]

    def _load_embedding_model(self):
        from sentence_transformers import SentenceTransformer

        kwargs = {}
        if revision := self.embedding_config.get("revision"):
            kwargs["revision"] = revision
        return SentenceTransformer(self.embedding_config["model_id"], **kwargs)

    def _load_reranker_model(self):
        from sentence_transformers import CrossEncoder

        kwargs = {}
        if revision := self.reranker_config.get("revision"):
            kwargs["revision"] = revision
        return CrossEncoder(self.reranker_config["model_id"], **kwargs)

    def load_enabled(self) -> None:
        for name, loader in (
            ("embeddings", self._load_embedding_model),
            ("reranker", self._load_reranker_model),
        ):
            if not self.enabled(name):
                continue
            try:
                self._models[name] = loader()
                self._load_errors.pop(name, None)
            except Exception as exc:
                self._models.pop(name, None)
                self._load_errors[name] = str(exc) or exc.__class__.__name__

    def is_available(self, name: str) -> bool:
        return self.enabled(name) and name in self._models

    def unavailable_reason(self, name: str) -> str:
        if not self.enabled(name):
            return f"{name.capitalize()} are disabled."
        return f"{name.capitalize()} are unavailable because the model failed to load."

    @property
    def embedding_model(self):
        return self._models["embeddings"]

    @property
    def reranker_model(self):
        return self._models["reranker"]

    def readiness(self) -> dict[str, dict[str, Any]]:
        return {
            "embeddings": self._model_state("embeddings", self.embedding_config),
            "reranker": self._model_state("reranker", self.reranker_config),
        }

    def _model_state(self, name: str, config: dict[str, Any]) -> dict[str, Any]:
        state: dict[str, Any] = {
            "enabled": self.enabled(name),
            "loaded": name in self._models,
            "model_id": config["model_id"],
            "revision": config.get("revision"),
        }
        if "dimensions" in config:
            state["dimensions"] = config["dimensions"]
        if name in self._load_errors:
            state["error"] = self._load_errors[name]
        return state
