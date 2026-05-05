# Eval Examples

Committed smoke-test fixtures for the evaluation suites.

These JSONL files provide small, reviewable cases for recommendation ranking, RAG retrieval, add-job extraction, and rubric-based output checks.

## Contents

- `add_job_extraction.jsonl` contains add-job parsing cases.
- `rag_retrieval.jsonl` contains expected retrieval cases.
- `recommendation_ranking.jsonl` contains ranking examples.
- `rubric_outputs.jsonl` contains agent-output examples for rubric scoring.

Keep these fixtures lightweight and non-sensitive. Larger or private benchmark data should stay outside the package, such as under `data/evals/`.
