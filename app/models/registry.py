import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


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

        for name in ("embeddings", "reranker"):
            cls._section_config(config, name)
            if not config[name]["enabled"]:
                continue

            model_config = cls._active_model_config(config, name)
            cls._require_non_empty_string(model_config, "model_id", name)
            if name == "embeddings":
                cls._require_positive_int(model_config, "dimensions", name)
            else:
                cls._require_positive_int(model_config, "return_top_k", name)
                if "rerank_top_k" in model_config:
                    cls._require_positive_int(model_config, "rerank_top_k", name)

    @staticmethod
    def _section_config(config: dict[str, Any], section_name: str) -> dict[str, Any]:
        section = config.get(section_name)
        if not isinstance(section, dict):
            raise ModelConfigError(f"Missing or invalid '{section_name}' config section.")
        if type(section.get("enabled")) is not bool:
            raise ModelConfigError(f"Missing or invalid 'enabled' for '{section_name}'.")
        return section

    @staticmethod
    def _active_model_config(config: dict[str, Any], section_name: str) -> dict[str, Any]:
        section = ModelRegistry._section_config(config, section_name)

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
        return self._active_model_config(self.config, "embeddings")

    @property
    def reranker_config(self) -> dict[str, Any]:
        return self._active_model_config(self.config, "reranker")

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
                logger.exception(
                    "Model failed to load",
                    extra={"model_type": name, "model_id": self._model_id(name)},
                )

    def is_available(self, name: str) -> bool:
        return self.enabled(name) and name in self._models

    def unavailable_reason(self, name: str) -> str:
        if not self.enabled(name):
            return f"{name.capitalize()} are disabled."
        return f"{name.capitalize()} are unavailable because the model failed to load."

    def readiness_status(self) -> str:
        if not self.is_available("embeddings"):
            return "not_ready"
        if self.enabled("reranker") and not self.is_available("reranker"):
            return "degraded"
        return "ready"

    @property
    def embedding_model(self):
        return self._models["embeddings"]

    @property
    def reranker_model(self):
        return self._models["reranker"]

    def readiness(self) -> dict[str, dict[str, Any]]:
        return {
            "embeddings": self._model_state("embeddings"),
            "reranker": self._model_state("reranker"),
        }

    def _model_state(self, name: str) -> dict[str, Any]:
        config = self._configured_model_if_available(name)
        state: dict[str, Any] = {
            "enabled": self.enabled(name),
            "loaded": name in self._models,
            "model_id": config.get("model_id") if config else None,
            "revision": config.get("revision") if config else None,
        }
        if config and "dimensions" in config:
            state["dimensions"] = config["dimensions"]
        if name in self._load_errors:
            state["error"] = "load_failed"
        return state

    def _configured_model_if_available(self, name: str) -> dict[str, Any] | None:
        section = self._section(name)
        active = section.get("active")
        models = section.get("models")
        if not isinstance(active, str) or not isinstance(models, dict):
            return None
        model_config = models.get(active)
        return model_config if isinstance(model_config, dict) else None

    def _model_id(self, name: str) -> str | None:
        config = self._configured_model_if_available(name)
        model_id = config.get("model_id") if config else None
        return model_id if isinstance(model_id, str) else None
