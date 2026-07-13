# Multilingual recommendation fixture

`multilingual_recommendations.json` is a repository-owned, deterministic, original-text fixture.
Each example has a language (`az`, `tr`, or `en`), a stable `query_id`, a query, candidate
objects, and `relevant_candidate_ids`. A listed ID is relevant to that query; every unlisted
candidate is a negative. Some negatives intentionally overlap vocabulary to make ranking less
trivial. The fixture is for regression measurement, not training or a claim of production quality.
