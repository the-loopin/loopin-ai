import pytest

from app.models.registry import ModelConfigError, ModelRegistry


VALID_CONFIG = {
    "embeddings": {
        "enabled": True,
        "active": "e5",
        "models": {
            "e5": {
                "model_id": "intfloat/multilingual-e5-small",
                "dimensions": 384,
            }
        },
    },
    "reranker": {
        "enabled": False,
        "active": "bge",
        "models": {
            "bge": {
                "model_id": "BAAI/bge-reranker-v2-m3",
                "return_top_k": 10,
            }
        },
    },
}


def test_registry_accepts_valid_config_without_loading_models():
    registry = ModelRegistry(VALID_CONFIG)

    assert registry.embedding_config["model_id"] == "intfloat/multilingual-e5-small"
    assert registry.reranker_config["model_id"] == "BAAI/bge-reranker-v2-m3"


def test_registry_rejects_missing_active_embedding_model():
    config = {
        **VALID_CONFIG,
        "embeddings": {
            "active": "missing",
            "models": VALID_CONFIG["embeddings"]["models"],
        },
    }

    with pytest.raises(ModelConfigError, match="Active model 'missing'"):
        ModelRegistry(config)


def test_registry_rejects_invalid_reranker_return_top_k():
    config = {
        **VALID_CONFIG,
        "reranker": {
            "active": "bge",
            "models": {
                "bge": {
                    "model_id": "BAAI/bge-reranker-v2-m3",
                    "return_top_k": 0,
                }
            },
        },
    }

    with pytest.raises(ModelConfigError, match="return_top_k"):
        ModelRegistry(config)


def test_registry_loads_only_enabled_models(monkeypatch):
    registry = ModelRegistry(VALID_CONFIG)
    embedding_model = object()
    monkeypatch.setattr(registry, "_load_embedding_model", lambda: embedding_model)
    monkeypatch.setattr(
        registry,
        "_load_reranker_model",
        lambda: pytest.fail("Disabled reranker must not be loaded."),
    )

    registry.load_enabled()

    assert registry.embedding_model is embedding_model
    assert registry.is_available("embeddings") is True
    assert registry.is_available("reranker") is False
    assert registry.readiness()["reranker"] == {
        "enabled": False,
        "loaded": False,
        "model_id": "BAAI/bge-reranker-v2-m3",
        "revision": None,
    }


def test_registry_keeps_healthy_model_available_when_other_model_fails(monkeypatch):
    config = {
        **VALID_CONFIG,
        "reranker": {**VALID_CONFIG["reranker"], "enabled": True},
    }
    registry = ModelRegistry(config)
    monkeypatch.setattr(registry, "_load_embedding_model", lambda: object())
    monkeypatch.setattr(
        registry,
        "_load_reranker_model",
        lambda: (_ for _ in ()).throw(RuntimeError("download failed")),
    )

    registry.load_enabled()

    assert registry.is_available("embeddings") is True
    assert registry.is_available("reranker") is False
    assert registry.readiness()["reranker"]["error"] == "download failed"


def test_registry_loads_enabled_reranker(monkeypatch):
    config = {
        **VALID_CONFIG,
        "reranker": {**VALID_CONFIG["reranker"], "enabled": True},
    }
    registry = ModelRegistry(config)
    reranker_model = object()
    monkeypatch.setattr(registry, "_load_embedding_model", lambda: object())
    monkeypatch.setattr(registry, "_load_reranker_model", lambda: reranker_model)

    registry.load_enabled()

    assert registry.reranker_model is reranker_model
    assert registry.is_available("reranker") is True
    assert registry.readiness()["reranker"]["loaded"] is True


def test_environment_can_enable_reranker_and_select_active_model(monkeypatch):
    config = {
        **VALID_CONFIG,
        "reranker": {
            **VALID_CONFIG["reranker"],
            "models": {
                **VALID_CONFIG["reranker"]["models"],
                "alternative": {
                    "model_id": "example/reranker",
                    "return_top_k": 5,
                },
            },
        },
    }
    monkeypatch.setenv("LOOPIN_RERANKER_ENABLED", "true")
    monkeypatch.setenv("LOOPIN_RERANKER_ACTIVE", "alternative")

    registry = ModelRegistry(ModelRegistry._with_environment_overrides(config))

    assert registry.enabled("reranker") is True
    assert registry.reranker_config["model_id"] == "example/reranker"
