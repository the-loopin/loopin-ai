import pytest

from app.models.registry import ModelConfigError, ModelRegistry


VALID_CONFIG = {
    "embeddings": {
        "active": "e5",
        "models": {
            "e5": {
                "model_id": "intfloat/multilingual-e5-small",
                "dimensions": 384,
            }
        },
    },
    "reranker": {
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
