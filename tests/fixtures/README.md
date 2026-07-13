# Multilingual recommendation fixture

`multilingual_recommendations.json` is a repository-owned, deterministic, original-text fixture.
Each example has a language (`az`, `tr`, or `en`), a stable `query_id`, a query, candidate
objects, and `relevant_candidate_ids`. A listed ID is relevant to that query; every unlisted
local candidate is a negative. `shared_corpus_relevance_labels` is used by the benchmark's shared
multilingual corpus and includes valid cross-language equivalents, so those matches are not false
negatives. Some negatives intentionally overlap vocabulary to make ranking less trivial. The
fixture is for regression measurement, not training or a claim of production quality.
