# Evals

Framework-agnostic evaluation code for Career Copilot behavior.

The evals package defines datasets, metrics, rubrics, and suite runners that can benchmark recommendation ranking, RAG retrieval, add-job extraction, and agent output quality.

## Contents

- `run.py` is the main evaluation runner.
- `datasets.py` loads committed or local evaluation data.
- `metrics.py` contains ranking and retrieval metrics.
- `rubrics.py` defines scored rubric checks for agent outputs.
- `examples/` stores small committed smoke-test fixtures.

Keep eval code independent of a specific orchestration framework so it can compare the current implementation with future agent rewrites.
