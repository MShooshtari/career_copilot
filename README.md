# Career Copilot

**AI Career Copilot** — A web app that helps users build a profile, get personalized job recommendations (RAG-backed), add or save jobs, improve their resume for specific roles, and prepare for interviews.

## Features

- **User profile** — Skills, experience, location, preferred roles, industries, work mode, salary range, and resume upload
- **Job recommendations (multi-stage)** — Candidate retrieval + model ranking + policy reranking:
  - **Candidate retrieval**: fetch a larger candidate pool using **pgvector** in PostgreSQL (e.g. top 100 by cosine distance to your profile embedding).
  - **Ranking**: score candidates with an optional MLflow-configured model using relevance, similarity, and freshness features.
  - **Policy reranking**: diversify the final window by penalizing jobs that look too similar to already selected jobs and reserve a small exploration slice for semi-random candidates.
  - **Tuning**: adjust recommendation knobs (candidate pool size, top-X window, default page sizes) in `src/career_copilot/constants.py`.
- **Job detail** — View full job description, source skills, extracted skills, and salary
- **Add job agent** — Paste a job URL or raw text; the agent extracts title, company, location, salary, description, and skills. Optional web search (Tavily or SerpAPI) fills in missing fields. Save the result to **My jobs**.
- **My jobs** — View, edit, and manage jobs you’ve added or saved from recommendations
- **Resume improvement agent** — For a chosen job: RAG-backed chat (similar jobs + similar resumes) and one-click PDF export of the improved resume. The agent can call tools to pull extra similar jobs/resumes from the vector store when it decides more context is useful.
- **Interview preparation agent** — Structured prep plan for a specific role and company. The agent can call a web-search tool (Glassdoor, Reddit, etc.) to fetch company-specific interview insights and weave them into the guidance.
- **Track applications** — A simple tracker where each “application” is either **resume improvement** or **interview preparation** for a specific job. Each application keeps **lightweight per-job memory** (last edited resume + compact summary) and only the **last N chat turns** for continuity.

## Tech stack

- **Backend:** FastAPI, PostgreSQL with **pgvector** (job and user-profile embeddings; HNSW indexes)
- **Embeddings:** OpenAI **text-embedding-3-small** (1536 dimensions; matches `EMBEDDING_VECTOR_DIMENSIONS` in `src/career_copilot/rag/embedding.py` and pgvector columns created by `init_schema`)
- **LLM:** OpenAI chat models with tool-calling for agentic behaviour (resume improvement, interview prep, add job)
- **Optional:** Tavily or SerpAPI for add-job agent web search
- **Web search (interview prep):** `ddgs` (DuckDuckGo search client)

## Prerequisites

- Python 3.11+
- PostgreSQL
- [OpenAI API key](https://platform.openai.com/api-keys) (for embeddings and all agents: resume improvement, interview prep, add job)

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment

Create a `.env` file in the project root (see `configs/config.example.env` for a minimal example):

```bash
# Database (required)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_DB=career_copilot
POSTGRES_PASSWORD=your_password

# Or a single DSN:
# POSTGRES_DSN=postgresql://user:password@localhost:5432/career_copilot

# OpenAI (required for embeddings and all agents)
OPENAI_API_KEY=sk-your-openai-key

# Optional: more jobs from Adzuna (https://developer.adzuna.com/signup)
# ADZUNA_APP_ID=your_app_id
# ADZUNA_APP_KEY=your_app_key

# Optional: add-job agent web search (fill missing fields from URL/text)
# TAVILY_API_KEY=your_tavily_key
# or SERPAPI_API_KEY=your_serpapi_key
```

### 3. Database and jobs

The web app creates **users**, **profiles**, **user_skills**, and **user_jobs** on startup. The **jobs** table (ingested listings) must exist before indexing; create it and run ingestion:

```bash
# Optional: run Postgres via Docker (image includes pgvector; see docker-compose.yml)
docker compose up -d

# Create jobs table (if not already present)
psql -d career_copilot -f sql/001_create_jobs.sql

# Incremental embedding updates (queue + trigger)
psql -d career_copilot -f sql/004_jobs_embedding_queue.sql

# Ingest jobs from RemoteOK, Remotive, Arbeitnow, (and Adzuna if configured).
# Ingestion also extracts normalized skill tags into jobs.extracted_skills.
python scripts/ingestion/run.py

# Optional: backfill extracted skills for jobs that already existed before this column was added
python scripts/job_skills_backfill/run.py

# Compute embeddings (two options):
# 1) One-shot backfill: Postgres jobs → embeddings in jobs_embeddings (requires OPENAI_API_KEY)
python scripts/job_embeddings_backfill/run.py
#
# 2) Incremental mode (recommended for online): DB trigger enqueues changes; run worker as a job
python scripts/job_embeddings_worker/run.py
```

#### Smoke test the incremental embedding queue (optional)

After applying `sql/004_jobs_embedding_queue.sql`, any new job insert or job `description` update will enqueue work:

```sql
-- Insert a dummy job
INSERT INTO jobs (source, source_id, title, description)
VALUES ('manual', 'smoke-1', 'Smoke test', 'First description');

-- Queue should have 1 row
SELECT * FROM jobs_embedding_queue ORDER BY requested_at DESC LIMIT 5;

-- Update description (queues again via UPSERT; row stays one-per-job)
UPDATE jobs SET description = 'Updated description' WHERE source='manual' AND source_id='smoke-1';
SELECT * FROM jobs_embedding_queue ORDER BY requested_at DESC LIMIT 5;
```

Then run the worker (needs `OPENAI_API_KEY`) and confirm the embedding upsert:

```bash
python scripts/job_embeddings_worker/run.py
psql -d career_copilot -c "SELECT count(*) FROM jobs_embeddings;"
```

### 4. Run the web app

```bash
python scripts/run_web.py
```

Then open **http://127.0.0.1:8000**. You’re redirected to `/profile`; fill in your profile and upload a resume. After saving, use **Recommendations** to (1) retrieve a candidate pool from the vector store and (2) rank it, showing only the **top X** results, **Add job** to parse a URL or paste and save to **My jobs**, **Resume improvement** for a chosen role, and **Interview preparation** for a structured prep plan.

You can also use **Track applications** (`/applications`) to see (and jump back into) your resume-improvement and interview-prep sessions. Sessions are stored per job/stage with a compact memory object and only the last N chat turns.

## ML experiments (MLflow)

This repo includes a **ranking baseline** with:

- **Label**: weak supervision from a latent ranking utility that combines *job-summary* / *resume-summary* similarity with freshness signals, then bins to `0`, `0.5`, or `1`.
- **Two dataset formats** per version (so you can use the right one per model):
  - **Similarity dataset** (`mock_similarity_vN.csv`): scalar features only — for **Logistic Regression, XGBoost**, etc.
  - **Embeddings dataset** (`mock_embeddings_vN.csv`): raw job and resume embedding dimensions + label — for **neural networks**.
- **Naming**: files are prefixed with `mock_` so that later you can add real user data under different names (e.g. `real_similarity_v1.csv`) and choose by name.
- **Class target**: training converts the weak labels to binary classes with `positive_threshold=1.0` by default, so only `1.0` is class `1`; `0.0` and `0.5` are class `0`.
- **Imbalance handling**: trainers use balanced sample weights and can optionally apply majority-class undersampling with `--undersample`. Each MLflow run logs whether undersampling was applied.

### Features (similarity dataset)

- `embedding_similarity`, `title_similarity`, `skill_overlap_count`, `location_match` (0/1), `experience_gap` (years)
- `salary_match`, `location_km` (distance in km), `days_since_posted`, `is_new` (posted within 3 days), `decay_score`
- `skill_similarity`, `role_similarity`, `work_mode_similarity`, `employment_type_similarity`, `preferred_locations_similarity` (intended to be computed from LLM-extracted tags in production; mock data simulates these)

At serving time, `embedding_similarity` is derived from pgvector distance and freshness features are computed from each job's `posted_at` metadata. Other scalar similarity fields are available for richer production feature engineering as those signals are added.

### Recommendation reranking

After model scoring, recommendations are reranked before pagination:

- Diversity: greedily selects jobs while penalizing candidates that overlap too much with already selected jobs by skills/title/category.
- Exploration: reserves `RECOMMENDATIONS_EXPLORATION_RATE` (default `0.20`) of the recommendation window for deterministic semi-random candidates from the retrieved pool.
- Tuning knobs live in `src/career_copilot/constants.py`: candidate pool size, rerank window size, diversity penalties, and exploration rate.

### Versioned dataset (single source of truth)

Data lives under `data/datasets/ranking/`. Each version writes two files: `mock_similarity_vN.csv` and `mock_embeddings_vN.csv`.

**Create a mock dataset version** (generates both similarity and embeddings CSVs):

```bash
PYTHONPATH=src python -m career_copilot.ml.create_ranking_dataset --n-rows 2000 --seed 7
# optional: --version v1 to name the version; otherwise v1, v2, ... auto-assigned
```

**Train (tree/linear)** using the **similarity** dataset:

```bash
PYTHONPATH=src python -m career_copilot.ml.train_logreg_mlflow --dataset-version latest
# or --dataset-version v1, v2, etc.
PYTHONPATH=src python -m career_copilot.ml.train_xgboost_mlflow --dataset-version latest --run-name xgboost-baseline
```

To compare imbalance strategies, run each trainer with and without `--undersample`.

For **neural networks**, load the **embeddings** dataset (e.g. `mock_embeddings_v1.csv`) in your training script; the current CLI trains only on the similarity dataset.

On Windows PowerShell (set `PYTHONPATH` once per terminal session, then run any of the commands below):

```powershell
$env:PYTHONPATH="src"
python -m career_copilot.ml.create_ranking_dataset --n-rows 2000 --seed 7
python -m career_copilot.ml.train_logreg_mlflow --dataset-version latest
python -m career_copilot.ml.train_logreg_mlflow --dataset-version latest --run-name logreg-undersampled --undersample
python -m career_copilot.ml.train_xgboost_mlflow --dataset-version latest --run-name xgboost-baseline
python -m career_copilot.ml.train_xgboost_mlflow --dataset-version latest --run-name xgboost-undersampled --undersample
```

If you see `ModuleNotFoundError: No module named 'career_copilot'`, run `$env:PYTHONPATH="src"` in the same terminal first, or install the package in editable mode from the repo root: `pip install -e .`

By default, runs are stored locally in `data/mlflow.db` and artifacts in `data/mlflow_artifacts`. To log to a remote MLflow server, set `MLFLOW_TRACKING_URI=https://<your-mlflow-server-url>` in `.env`. When using a remote MLflow server, artifact storage should be configured on that server; the local training client usually does not need `MLFLOW_EXPERIMENT_ARTIFACT_LOCATION`.

Datasets live in `data/datasets/ranking/` (`mock_similarity_vN.csv`, `mock_embeddings_vN.csv`, `manifest.json`). Create a new dataset version and retrain models whenever the feature contract changes.

### Run MLflow UI (view experiments)

Start the UI:

```bash
mlflow ui --backend-store-uri "sqlite:///./data/mlflow.db" --default-artifact-root "file:./data/mlflow_artifacts"
```

On **Windows**, use a single worker to avoid `OSError: [WinError 10022]` (uvicorn multiprocess socket issue):

```bash
mlflow ui --backend-store-uri "sqlite:///./data/mlflow.db" --default-artifact-root "file:./data/mlflow_artifacts" --workers 1
```

Then open `http://127.0.0.1:5000` and look for:

- **Experiments**: `career-copilot-ranking` (default name)
- **Runs**: each training run logs params + metrics
- **Artifacts**:
  - `model/` (the trained Logistic Regression pipeline)
  - `eval/confusion_matrix.csv`

## Job sources (ingestion)

| Source      | API key | Notes                    |
|------------|--------|---------------------------|
| RemoteOK   | No     | ~100 jobs per run         |
| Remotive   | No     | Remote jobs               |
| Arbeitnow  | No     | Europe-focused            |
| Adzuna     | Yes    | Set `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` in `.env` for more jobs |

Ingestion stores both source-provided `skills` and dynamically extracted `extracted_skills` from job descriptions. Existing rows can be backfilled with `python scripts/job_skills_backfill/run.py`; new rows are populated by `python scripts/ingestion/run.py`.

To refresh jobs on a schedule, run `python scripts/ingestion/scheduler.py` (default: every 6 hours; edit the script to change the interval). Re-run `python scripts/job_embeddings_backfill/run.py` after ingestion to refresh **job embeddings** in Postgres (pgvector).

## Project structure

```
career_copilot/
├── src/career_copilot/
│   ├── web_app.py          # FastAPI app, routers
│   ├── app_config.py       # Paths, Jinja2 templates
│   ├── schemas.py          # Pydantic request models
│   ├── utils.py            # Shared helpers
│   ├── routers/            # home, profile, jobs, recommendations, add_job, my_jobs, resume_improvement, interview_preparation, track_applications
│   ├── database/           # db, schema, profiles, jobs, applications, deps
│   ├── rag/                # pgvector RAG (pgvector_rag, embedding, job_document)
│   ├── ml/                 # Ranking datasets (ranking_dataset, dataset_store), create_ranking_dataset, train_logreg_mlflow
│   ├── ingestion/          # Job APIs (RemoteOK, Remotive, Arbeitnow, Adzuna) and skill extraction
│   ├── agents/             # resume_improvement, interview_preparation, add_job, track_applications, application_memory
│   ├── resume_io.py        # Resume text extraction (PDF)
│   └── resume_pdf.py       # PDF generation for improved resume
├── templates/              # Jinja2 HTML (profile, recommendations, job_detail, improve_resume, add_job, my_jobs, interview_prep)
├── configs/
│   └── config.example.env  # Example .env (use project root .env)
├── tests/                  # Pytest unit tests
├── scripts/
│   ├── ingestion/
│   │   ├── Dockerfile      # Scheduled ingestion image (build from repo root; see file header)
│   │   ├── requirements.txt # Pip deps for that image only
│   │   ├── run.py          # Fetch jobs → Postgres
│   │   └── scheduler.py    # Optional: run ingestion on a schedule (e.g. every 6 hours)
│   ├── job_embeddings_backfill/
│   │   ├── Dockerfile      # One-shot backfill image (build from repo root; see file header)
│   │   ├── requirements.txt # Pip deps for that image only
│   │   └── run.py          # Postgres jobs → embeddings in jobs_embeddings
│   ├── job_embeddings_worker/
│   │   ├── Dockerfile      # Incremental worker image (drains queue → upserts embeddings)
│   │   ├── requirements.txt # Pip deps for that image only
│   │   └── run.py          # Drain jobs_embedding_queue → upsert jobs_embeddings
│   ├── job_skills_backfill/
│   │   └── run.py          # Existing jobs → extracted skill tags in jobs.extracted_skills
│   ├── run_web.py          # Start uvicorn
│   ├── repair_descriptions.py
│   └── explore_embeddings.py
├── sql/                    # 001_create_jobs.sql, 002_create_user_jobs.sql, 003_pgvector.sql (vectors also in init_schema)
├── docker-compose.yml      # Postgres 16 for local dev
├── pyproject.toml          # Ruff, pytest config
├── requirements.txt
└── requirements-dev.txt    # Ruff, pip-audit
```

## Development

### Tests

```bash
# From project root (pyproject.toml sets pythonpath)
pytest tests/ -v

# Or with PYTHONPATH
PYTHONPATH=src pytest tests/ -v
```

There are unit tests for RAG job-document helpers (`tests/test_job_document.py`), dynamic skill extraction (`tests/test_skill_extraction.py`), and a small contract test that pgvector DDL in `init_schema` still interpolates `EMBEDDING_VECTOR_DIMENSIONS` (`tests/test_embedding_schema_contract.py`), so schema and embedding config stay aligned.

### Linting and formatting

```bash
pip install -r requirements-dev.txt

ruff check src tests
ruff format --check src tests

# Auto-fix and format
ruff check src tests --fix
ruff format src tests
```

### CI

The **pr-checks** GitHub Action runs on pull requests and pushes to `main`/`master`:

- **Lint** — Ruff check + format check
- **Test** — Pytest on Python 3.11 and 3.12
- **Security** — `pip-audit` on dependencies

To **block merging until all checks pass**, configure [branch protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches) and require the pr-checks status checks. See [.github/REQUIRE_CHECKS_BEFORE_MERGE.md](.github/REQUIRE_CHECKS_BEFORE_MERGE.md) for step-by-step instructions.

## License

See [LICENSE](LICENSE) (if present).
