from app.services.embedding_service import EmbeddingService
from app.services.reranker_service import RerankerService


class FakeEmbeddingModel:
    def encode(self, texts, normalize_embeddings=True):
        assert texts[0].startswith("query: ")
        return FakeVector([[1.0, 0.0, 0.0]])


class FakeRerankerModel:
    def predict(self, pairs):
        return [0.1, 0.9]


class FakeVector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeRegistry:
    embedding_config = {
        "model_id": "intfloat/multilingual-e5-small",
        "dimensions": 384,
        "normalize": True,
    }
    reranker_config = {
        "model_id": "BAAI/bge-reranker-v2-m3",
        "return_top_k": 10,
    }
    embedding_model = FakeEmbeddingModel()
    reranker_model = FakeRerankerModel()


def test_embedding_service_prefixes_query_text():
    result = EmbeddingService(FakeRegistry()).embed(["music"], input_type="query")

    assert result.model == "intfloat/multilingual-e5-small"
    assert result.dimensions == 384
    assert result.embeddings == [[1.0, 0.0, 0.0]]


def test_reranker_service_returns_highest_score_first():
    result = RerankerService(FakeRegistry()).rerank(
        query="music",
        candidates=[
            {"id": "event_1", "text": "coding meetup"},
            {"id": "event_2", "text": "jazz night"},
        ],
    )

    assert [item["id"] for item in result.results] == ["event_2", "event_1"]
