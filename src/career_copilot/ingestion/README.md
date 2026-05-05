# Ingestion

Job ingestion clients and normalization helpers.

This package fetches listings from external job sources, normalizes shared fields, and extracts skill tags before jobs are stored or indexed.

## Contents

- `remoteok_api.py`, `remotive_api.py`, `arbeitnow_api.py`, and `adzuna_api.py` fetch jobs from supported providers.
- `common.py` contains shared normalization helpers.
- `skill_extraction.py` extracts normalized skills from job content.

Keep provider-specific behavior isolated in its own module and route shared cleanup through `common.py`.
