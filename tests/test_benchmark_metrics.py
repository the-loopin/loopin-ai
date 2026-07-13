from benchmarks.metrics import query_metrics, rank_by_score, summarize_queries


def test_rank_by_score_uses_candidate_id_for_equal_scores():
    assert rank_by_score([("z", 0.5), ("a", 0.5), ("b", 0.9)]) == ["b", "a", "z"]


def test_metrics_compute_recall_and_first_relevant_reciprocal_rank():
    row = query_metrics(["negative", "relevant", "other"], {"relevant"})
    assert row == {"recall_at_10": 1.0, "reciprocal_rank": 0.5}
    summary = summarize_queries([
        {"language": "en", "relevant_candidates": 1, **row},
        {"language": "tr", "relevant_candidates": 2, **query_metrics(["negative"], {"relevant"})},
    ])
    assert summary["overall"] == {
        "evaluated_queries": 2,
        "relevant_candidates": 3,
        "recall_at_10": 0.5,
        "mrr": 0.25,
    }
