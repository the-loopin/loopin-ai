"""Ranking metrics used by the recommendation benchmark.

Recall@k is the fraction of evaluated queries with at least one relevant candidate in their
first k results. MRR is the mean reciprocal rank: for each query it contributes 1/rank of its
first relevant candidate, or zero when no relevant candidate is retrieved.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def rank_by_score(items: Iterable[tuple[str, float]]) -> list[str]:
    """Rank descending by score, breaking ties by stable candidate ID."""
    return [candidate_id for candidate_id, _ in sorted(items, key=lambda item: (-item[1], item[0]))]


def query_metrics(ranked_ids: list[str], relevant_ids: set[str], k: int = 10) -> dict[str, float]:
    top_k = ranked_ids[:k]
    recall = float(bool(set(top_k) & relevant_ids))
    reciprocal_rank = 0.0
    for index, candidate_id in enumerate(ranked_ids, start=1):
        if candidate_id in relevant_ids:
            reciprocal_rank = 1.0 / index
            break
    return {"recall_at_10": recall, "reciprocal_rank": reciprocal_rank}


def summarize_queries(rows: list[dict]) -> dict:
    """Summarize already-scored query rows overall and by language."""
    groups: dict[str, list[dict]] = defaultdict(list)
    groups["overall"] = rows
    for row in rows:
        groups[row["language"]].append(row)

    def summarize(group: list[dict]) -> dict:
        count = len(group)
        return {
            "evaluated_queries": count,
            "relevant_candidates": sum(row["relevant_candidates"] for row in group),
            "recall_at_10": sum(row["recall_at_10"] for row in group) / count if count else 0.0,
            "mrr": sum(row["reciprocal_rank"] for row in group) / count if count else 0.0,
        }

    return {language: summarize(group) for language, group in groups.items()}
