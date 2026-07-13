"""Run the opt-in real-model recommendation benchmark.

This script never records query or candidate text in its outputs. It reads model IDs,
dimensions, active selections, and optional revisions from config/models.yaml through
ModelRegistry. Model download time is recorded separately from warm steady-state inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Callable, TypeVar

import psutil

# Running this file directly makes ``benchmarks/`` the first import path. Add the
# repository root so the documented ``python benchmarks/run_recommendation_benchmark.py`` works.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.registry import ModelRegistry
from app.runtime import InferenceRuntimeSettings
from app.services.embedding_service import EmbeddingService
from app.services.reranker_service import RerankerService
from benchmarks.metrics import query_metrics, rank_by_score, summarize_queries


FIXTURE_PATH = ROOT / "tests" / "fixtures" / "multilingual_recommendations.json"
RESULTS_DIR = ROOT / "benchmark-results"
LATEST_REPORT = ROOT / "docs" / "benchmarks" / "latest.md"
T = TypeVar("T")


def _rss_mib(process: psutil.Process) -> float:
    return round(process.memory_info().rss / (1024 * 1024), 2)


def _safe_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _resolved_commit(model: object) -> str | None:
    """Return an HF commit recorded by transformers, without making another network call."""
    possible = [model, getattr(model, "model", None)]
    try:
        possible.extend([model[0], model._first_module(), getattr(model._first_module(), "auto_model", None)])
    except (AttributeError, IndexError, KeyError, TypeError):
        pass
    for item in possible:
        config = getattr(item, "config", None)
        commit = getattr(config, "_commit_hash", None)
        if isinstance(commit, str) and commit:
            return commit
    return None


def _timed_runs(operation: Callable[[], T], *, warmups: int, runs: int, items: int, process: psutil.Process) -> tuple[T, dict, float]:
    for _ in range(warmups):
        operation()
    samples: list[float] = []
    peak_rss = _rss_mib(process)
    result: T | None = None
    for _ in range(runs):
        started = perf_counter()
        result = operation()
        samples.append(perf_counter() - started)
        peak_rss = max(peak_rss, _rss_mib(process))
    ordered = sorted(samples)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return result, {
        "measured_runs": runs,
        "warmup_runs_excluded": warmups,
        "median_ms": round(statistics.median(samples) * 1000, 3),
        "p95_ms": round(ordered[p95_index] * 1000, 3),
        "min_ms": round(min(samples) * 1000, 3),
        "max_ms": round(max(samples) * 1000, 3),
        "items_per_second": round((items * runs) / sum(samples), 3),
    }, peak_rss


def _cosine(left: list[float], right: list[float]) -> float:
    # Embeddings are normalized by configuration, but calculate the full cosine for safety.
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    return sum(x * y for x, y in zip(left, right, strict=True)) / denominator if denominator else 0.0


def _dataset() -> tuple[dict, str]:
    raw = FIXTURE_PATH.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _quality_rows(dataset: dict, embedder: EmbeddingService, reranker: RerankerService | None) -> tuple[list[dict], list[dict] | None]:
    vector_rows: list[dict] = []
    reranked_rows: list[dict] | None = [] if reranker else None
    # Rank against the fixture-wide corpus, so Recall@10 is meaningful rather than
    # trivially perfect against the four candidates shown beside each query.
    corpus = [candidate for example in dataset["examples"] for candidate in example["candidates"]]
    candidate_by_id = {candidate["id"]: candidate for candidate in corpus}
    assert len(candidate_by_id) == len(corpus), "fixture candidate IDs must be globally unique"
    passage_vectors = embedder.embed([candidate["text"] for candidate in corpus], input_type="passage").embeddings
    for example in dataset["examples"]:
        query = embedder.embed([example["query"]], input_type="query").embeddings[0]
        ranked = rank_by_score((candidate["id"], _cosine(query, vector)) for candidate, vector in zip(corpus, passage_vectors, strict=True))
        relevant = set(example["relevant_candidate_ids"])
        vector_rows.append({
            "query_id": example["query_id"], "language": example["language"],
            "relevant_candidates": len(relevant), "ranked_candidate_ids": ranked,
            **query_metrics(ranked, relevant),
        })
        if reranker and reranked_rows is not None:
            pool_ids = ranked[: min(50, len(ranked))]
            reranked = reranker.rerank(example["query"], [candidate_by_id[item] for item in pool_ids], top_k=10)
            reranked_ids = [item["id"] for item in reranked.results]
            reranked_rows.append({
                "query_id": example["query_id"], "language": example["language"],
                "relevant_candidates": len(relevant), "ranked_candidate_ids": reranked_ids,
                **query_metrics(reranked_ids, relevant),
            })
    return vector_rows, reranked_rows


def _comparison(vector_rows: list[dict], reranked_rows: list[dict] | None) -> dict | None:
    if reranked_rows is None:
        return None
    vector = summarize_queries(vector_rows)
    reranked = summarize_queries(reranked_rows)
    comparison: dict[str, dict] = {}
    for language in vector:
        item: dict[str, float | int | None] = {}
        for metric in ("recall_at_10", "mrr"):
            before, after = vector[language][metric], reranked[language][metric]
            item[f"vector_{metric}"] = before
            item[f"reranked_{metric}"] = after
            item[f"absolute_{metric}_change"] = after - before
            item[f"percent_{metric}_change"] = ((after - before) / before * 100) if before else None
        comparison[language] = item
    outcomes = {"improved": 0, "unchanged": 0, "worse": 0}
    for before, after in zip(vector_rows, reranked_rows, strict=True):
        delta = after["reciprocal_rank"] - before["reciprocal_rank"]
        outcomes["improved" if delta > 0 else "worse" if delta < 0 else "unchanged"] += 1
    return {"by_language": comparison, "query_outcomes": outcomes}


def _latency(embedder: EmbeddingService, reranker: RerankerService | None, dataset: dict, process: psutil.Process, warmups: int, runs: int) -> tuple[dict, float, float | None]:
    sample = dataset["examples"][0]
    base_texts = [candidate["text"] for candidate in sample["candidates"]]
    embedding_peak = _rss_mib(process)
    embedding: dict[str, dict] = {}
    for label, count, input_type in (("single_query", 1, "query"), ("batch_8", 8, "passage"), ("batch_32", 32, "passage")):
        texts = (base_texts * math.ceil(count / len(base_texts)))[:count]
        _, result, scenario_peak = _timed_runs(lambda: embedder.embed(texts, input_type=input_type), warmups=warmups, runs=runs, items=count, process=process)
        embedding[label] = result
        embedding_peak = max(embedding_peak, scenario_peak)
    reranking: dict[str, dict] = {}
    reranker_peak: float | None = None
    if reranker:
        for count in (5, 10, 20, 50):
            candidates = [
                {"id": f"latency_{index:02d}", "text": base_texts[index % len(base_texts)], "metadata": {"synthetic": True}}
                for index in range(count)
            ]
            _, result, scenario_peak = _timed_runs(lambda: reranker.rerank(sample["query"], candidates, top_k=10), warmups=warmups, runs=runs, items=count, process=process)
            reranking[str(count)] = result
            reranker_peak = max(reranker_peak or scenario_peak, scenario_peak)
    return {"embedding_inference": embedding, "reranker_inference": reranking}, embedding_peak, reranker_peak


def _recommendation(comparison: dict | None, latency: dict, peak_memory_delta_mib: float) -> tuple[str, str]:
    if comparison is None:
        return "More evidence required", "The reranker was not benchmarked; production remains disabled."
    overall = comparison["by_language"]["overall"]
    if overall["absolute_recall_at_10_change"] <= 0 and overall["absolute_mrr_change"] <= 0:
        return "Keep reranker disabled", "The measured fixture shows no ranking-quality gain, so additional latency and RSS cost is not justified."
    reranker_p95 = latency["reranker_inference"]["50"]["p95_ms"]
    return "More evidence required", (
        "The fixture shows a quality gain, but the measured reranker p95 at 50 candidates "
        f"is {reranker_p95:.3f} ms and peak process RSS increased by {peak_memory_delta_mib:.2f} MiB. "
        "The small 15-query regression fixture is not representative enough alone to justify enabling it."
    )


def _markdown(result: dict) -> str:
    quality = result["quality"]["vector_only"]
    lines = [
        "# Latest recommendation benchmark",
        "",
        f"Generated: `{result['timestamp']}`. Results contain IDs and aggregates only; query and candidate text are intentionally omitted.",
        "",
        "## Environment and models",
        "",
        f"- Embedding: `{result['models']['embedding']['model_id']}` (configured revision: `{result['models']['embedding']['configured_revision'] or 'absent'}`, resolved commit: `{result['models']['embedding']['resolved_commit'] or 'unavailable'}`)",
        f"- Reranker: `{result['models']['reranker']['model_id']}` (configured revision: `{result['models']['reranker']['configured_revision'] or 'absent'}`, resolved commit: `{result['models']['reranker']['resolved_commit'] or 'unavailable'}`)",
        f"- Embedding dimensions: {result['models']['embedding']['dimensions']}; fixture SHA-256: `{result['dataset']['sha256']}`",
        f"- OS: `{result['environment']['operating_system']}`; CPU: `{result['environment']['cpu']}`; Python: `{result['environment']['python']}`",
        "",
        "## Vector-only quality",
        "",
        "| Language | Queries | Relevant candidates | Recall@10 | MRR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for language in ("overall", "az", "tr", "en"):
        row = quality[language]
        lines.append(f"| {language} | {row['evaluated_queries']} | {row['relevant_candidates']} | {row['recall_at_10']:.4f} | {row['mrr']:.4f} |")
    comparison = result["quality"].get("reranker_comparison")
    if comparison:
        lines.extend(["", "## Reranker comparison", "", "| Language | Vector Recall@10 | Reranked Recall@10 | Δ Recall@10 | Vector MRR | Reranked MRR | Δ MRR |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for language in ("overall", "az", "tr", "en"):
            row = comparison["by_language"][language]
            lines.append(f"| {language} | {row['vector_recall_at_10']:.4f} | {row['reranked_recall_at_10']:.4f} | {row['absolute_recall_at_10_change']:+.4f} | {row['vector_mrr']:.4f} | {row['reranked_mrr']:.4f} | {row['absolute_mrr_change']:+.4f} |")
        lines.append(f"\nQuery outcomes: {comparison['query_outcomes']}.")
    lines.extend([
        "", "## Latency and process RSS", "",
        "Load durations exclude downloads from steady-state inference measurements. RSS values are process RSS in MiB; peak is the highest sample observed before/after each measured inference.",
        "",
        f"- Embedding model load: {result['models']['embedding']['load_duration_seconds']:.3f} s; reranker model load: {result['models']['reranker']['load_duration_seconds']!s} s.",
        "",
        "| Operation | Runs | Median ms | P95 ms | Min ms | Max ms | Items/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name, row in result["latency"]["embedding_inference"].items():
        lines.append(f"| embedding {name} | {row['measured_runs']} | {row['median_ms']:.3f} | {row['p95_ms']:.3f} | {row['min_ms']:.3f} | {row['max_ms']:.3f} | {row['items_per_second']:.3f} |")
    for count, row in result["latency"]["reranker_inference"].items():
        lines.append(f"| reranker {count} candidates | {row['measured_runs']} | {row['median_ms']:.3f} | {row['p95_ms']:.3f} | {row['min_ms']:.3f} | {row['max_ms']:.3f} | {row['items_per_second']:.3f} |")
    lines.extend([
        "",
        f"- Baseline RSS: {result['memory']['baseline_rss_mib']:.2f} MiB; after embedding load: {result['memory']['after_embedding_load_rss_mib']:.2f} MiB; peak embedding: {result['memory']['peak_embedding_rss_mib']:.2f} MiB.",
        f"- Reranker after-load RSS: {result['memory'].get('after_reranker_load_rss_mib')}; peak reranker RSS: {result['memory'].get('peak_reranker_rss_mib')}; peak delta: {result['memory']['peak_delta_from_baseline_mib']:.2f} MiB.",
        "", "## Recommendation", "", f"**{result['recommendation']['decision']}** — {result['recommendation']['reason']}",
        "", "## Limitations", "", "This is a small deterministic regression fixture (five queries per language), not a production relevance study. RSS sampling measures the process and may miss sub-millisecond native allocation peaks.", "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark configured Loopin recommendation models.")
    parser.add_argument("--include-reranker", action="store_true", help="load and compare the configured reranker (also requires RUN_RERANKER_TESTS=true)")
    parser.add_argument("--runs", type=int, default=5, help="measured runs per latency scenario")
    parser.add_argument("--warmups", type=int, default=1, help="warm-up runs excluded from latency results")
    args = parser.parse_args()
    if args.runs < 1 or args.warmups < 0:
        parser.error("--runs must be positive and --warmups cannot be negative")
    if args.include_reranker and os.getenv("RUN_RERANKER_TESTS", "").lower() != "true":
        parser.error("--include-reranker requires RUN_RERANKER_TESTS=true")

    old_reranker = os.environ.get("LOOPIN_RERANKER_ENABLED")
    # Load embedding first, then the optional reranker, so after-load RSS is incremental.
    os.environ["LOOPIN_RERANKER_ENABLED"] = "false"
    runtime_settings = InferenceRuntimeSettings.from_environment()
    runtime_settings.apply_cpu_environment()
    process = psutil.Process()
    baseline_rss = _rss_mib(process)
    try:
        registry = ModelRegistry.from_yaml(str(ROOT / "config" / "models.yaml"))
        registry.load_enabled()
        if not registry.is_available("embeddings"):
            raise RuntimeError("Configured embedding model failed to load; check cache, network access, and model-load logs.")
        after_embedding = _rss_mib(process)
        after_reranker = None
        if args.include_reranker:
            registry.config["reranker"]["enabled"] = True
            started = perf_counter()
            try:
                registry._models["reranker"] = registry._load_reranker_model()
            except Exception as exc:
                registry._models.pop("reranker", None)
                registry._load_errors["reranker"] = str(exc) or exc.__class__.__name__
                raise RuntimeError("Configured reranker failed to load; check cache, network access, and model-load logs.") from exc
            finally:
                registry._load_durations["reranker"] = perf_counter() - started
            after_reranker = _rss_mib(process)
        dataset, fixture_hash = _dataset()
        embedder = EmbeddingService(registry)
        reranker = RerankerService(registry) if args.include_reranker else None
        vector_rows, reranked_rows = _quality_rows(dataset, embedder, reranker)
        latency, peak_embedding, peak_reranker = _latency(embedder, reranker, dataset, process, args.warmups, args.runs)
        comparison = _comparison(vector_rows, reranked_rows)
        peak_delta = round(max(peak_embedding, peak_reranker or 0) - baseline_rss, 2)
        decision, reason = _recommendation(comparison, latency, peak_delta)
        result = {
            "timestamp": datetime.now(UTC).isoformat(),
            "dataset": {"version": dataset["dataset_version"], "path": str(FIXTURE_PATH.relative_to(ROOT)), "sha256": fixture_hash},
            "models": {
                "embedding": {"model_id": registry.embedding_config["model_id"], "configured_revision": registry.embedding_config.get("revision"), "resolved_commit": _resolved_commit(registry.embedding_model), "dimensions": registry.embedding_config["dimensions"], "load_duration_seconds": registry.load_duration("embeddings")},
                "reranker": {"model_id": registry.reranker_config["model_id"], "configured_revision": registry.reranker_config.get("revision"), "resolved_commit": _resolved_commit(registry.reranker_model) if reranker else None, "enabled_for_run": bool(reranker), "load_duration_seconds": registry.load_duration("reranker") if reranker else None},
            },
            "environment": {"git_commit": _git_sha(), "python": sys.version, "operating_system": platform.platform(), "cpu": platform.processor() or platform.machine(), "logical_cpu_count": psutil.cpu_count(), "cpu_thread_settings": {"omp_num_threads": runtime_settings.omp_num_threads, "mkl_num_threads": runtime_settings.mkl_num_threads, "tokenizers_parallelism": runtime_settings.tokenizers_parallelism}, "packages": {name: _safe_version(name) for name in ("sentence-transformers", "torch", "transformers", "psutil")}},
            "quality": {"vector_only": summarize_queries(vector_rows), "reranker": summarize_queries(reranked_rows) if reranked_rows is not None else None, "reranker_comparison": comparison, "query_rows": {"vector_only": vector_rows, "reranker": reranked_rows}},
            "latency": latency,
            "memory": {"baseline_rss_mib": baseline_rss, "after_embedding_load_rss_mib": after_embedding, "after_reranker_load_rss_mib": after_reranker, "peak_embedding_rss_mib": peak_embedding, "peak_reranker_rss_mib": peak_reranker, "peak_delta_from_baseline_mib": peak_delta},
            "recommendation": {"decision": decision, "reason": reason},
        }
        RESULTS_DIR.mkdir(exist_ok=True)
        output_path = RESULTS_DIR / f"recommendation-benchmark-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        LATEST_REPORT.parent.mkdir(parents=True, exist_ok=True)
        LATEST_REPORT.write_text(_markdown(result), encoding="utf-8")
        print(f"Benchmark JSON: {output_path.relative_to(ROOT)}")
        print(f"Markdown report: {LATEST_REPORT.relative_to(ROOT)}")
        print(f"Recommendation: {decision}")
        return 0
    finally:
        if old_reranker is None:
            os.environ.pop("LOOPIN_RERANKER_ENABLED", None)
        else:
            os.environ["LOOPIN_RERANKER_ENABLED"] = old_reranker


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Benchmark blocked: {exc}", file=sys.stderr)
        raise SystemExit(2)
