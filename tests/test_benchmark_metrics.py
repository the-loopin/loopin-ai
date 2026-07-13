import json
from pathlib import Path

from benchmarks.metrics import query_metrics, rank_by_score, summarize_queries
from benchmarks.run_recommendation_benchmark import _recommendation


def test_rank_by_score_uses_candidate_id_for_equal_scores():
    assert rank_by_score([("z", 0.5), ("a", 0.5), ("b", 0.9)]) == ["b", "a", "z"]


def test_metrics_compute_recall_and_first_relevant_reciprocal_rank():
    row = query_metrics(["negative", "relevant", "other"], {"relevant"})
    assert row == {"recall_at_10": 1.0, "reciprocal_rank": 0.5}
    partial_row = query_metrics(["relevant_1"], {"relevant_1", "relevant_2"})
    summary = summarize_queries([
        {"language": "en", "relevant_candidates": 1, **row},
        {"language": "tr", "relevant_candidates": 2, **partial_row},
    ])
    assert summary["overall"] == {
        "evaluated_queries": 2,
        "relevant_candidates": 3,
        "recall_at_10": 0.75,
        "mrr": 0.75,
    }


def test_recall_at_10_counts_all_relevant_candidates_not_just_any_hit():
    row = query_metrics(
        ["relevant_1", "negative", "relevant_2", "other"],
        {"relevant_1", "relevant_2", "relevant_3"},
    )

    assert row == {"recall_at_10": 2 / 3, "reciprocal_rank": 1.0}


def test_recall_at_10_respects_the_cutoff_for_multiple_relevant_candidates():
    ranked = [f"negative_{index}" for index in range(9)] + ["relevant_1", "relevant_2"]

    assert query_metrics(ranked, {"relevant_1", "relevant_2"}) == {
        "recall_at_10": 0.5,
        "reciprocal_rank": 0.1,
    }


def test_shared_corpus_labels_cover_cross_language_relevance_without_unknown_ids():
    fixture_path = Path("tests/fixtures/multilingual_recommendations.json")
    dataset = json.loads(fixture_path.read_text(encoding="utf-8"))
    candidate_ids = {
        candidate["id"]
        for example in dataset["examples"]
        for candidate in example["candidates"]
    }

    assert set(dataset["shared_corpus_relevance_labels"]) == {
        example["query_id"] for example in dataset["examples"]
    }
    assert all(
        set(relevant_ids) <= candidate_ids and len(relevant_ids) >= 3
        for relevant_ids in dataset["shared_corpus_relevance_labels"].values()
    )


def _comparison(recall_gain: float, mrr_gain: float) -> dict:
    return {
        "by_language": {
            "overall": {
                "absolute_recall_at_10_change": recall_gain,
                "absolute_mrr_change": mrr_gain,
            }
        }
    }


def test_recommendation_can_enable_reranker_within_explicit_thresholds():
    decision, _ = _recommendation(
        _comparison(0.03, 0.04),
        {"reranker_inference": {"50": {"p95_ms": 700.0}}},
        700.0,
    )

    assert decision == "Enable reranker"


def test_recommendation_disables_reranker_for_quality_or_cost_regressions():
    assert _recommendation(
        _comparison(0.0, -0.01),
        {"reranker_inference": {"50": {"p95_ms": 700.0}}},
        700.0,
    )[0] == "Keep reranker disabled"
    assert _recommendation(
        _comparison(0.03, 0.04),
        {"reranker_inference": {"50": {"p95_ms": 2100.0}}},
        700.0,
    )[0] == "Keep reranker disabled"


def test_recommendation_requests_more_evidence_between_thresholds():
    assert _recommendation(
        _comparison(0.01, 0.03),
        {"reranker_inference": {"50": {"p95_ms": 700.0}}},
        700.0,
    )[0] == "More evidence required"
