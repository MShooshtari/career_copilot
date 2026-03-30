# Career Copilot

**AI Career Copilot** — A web app that helps users build a profile, get personalized job recommendations (RAG-backed), add or save jobs, improve their resume for specific roles, and prepare for interviews.

## Features

- **User profile** — Skills, experience, location, preferred roles, industries, work mode, salary range, and resume upload
- **Job recommendations (two-stage)** — Candidate retrieval + ranking:
  - **Candidate retrieval**: fetch a larger candidate pool using **pgvector** in PostgreSQL (e.g. top 100 by cosine distance to your profile embedding).
  - **Ranking**: re-rank those candidates (optionally via an MLflow-configured model) and **only show the top X** results (default **15**, paginated within that window).
  - **Tuning**: adjust recommendation knobs (candidate pool size, top-X window, default page sizes) in `src/career_copilot/constants.py`.
- **Job detail** — View full job description, skills, and salary
- **Add job agent** — Paste a job URL or raw text; the agent extracts title, company, location, salary, description, and skills. Optional web search (Tavily or SerpAPI) fills in missing fields. Save the result to **My jobs**.
- **My jobs** — View, edit, and manage jobs you’ve added or saved from recommendations
- **Resume improvement agent** — For a chosen job: RAG-backed chat (similar jobs + similar resumes) and one-click PDF export of the improved resume. The agent can call tools to pull extra similar jobs/resumes from the vector store when it decides more context is useful.
- **Interview preparation agent** — Structured prep plan for a specific role and company. The agent can call a web-search tool (Glassdoor, Reddit, etc.) to fetch company-specific interview insights and weave them into the guidance.
- **Track applications** — A simple tracker where each “application” is either **resume improvement** or **interview preparation** for a specific job. Each application keeps **lightweight per-job memory** (last edited resume + compact summary) and only the **last N chat turns** for continuity.

## Tech stack

- **Backend:** FastAPI, PostgreSQL with **pgvector** (job and user-profile embeddings; HNSW indexes)
- **Embeddings:** OpenAI text-embedding-3-large (jobs and user profiles)
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

# Ingest jobs from RemoteOK, Remotive, Arbeitnow, (and Adzuna if configured)
python scripts/ingestion/run.py

# Compute embeddings and store them in jobs_embeddings (pgvector) — requires OPENAI_API_KEY
python scripts/rag_index/run.py
```

### 4. Run the web app

```bash
python scripts/run_web.py
```

Then open **http://127.0.0.1:8000**. You’re redirected to `/profile`; fill in your profile and upload a resume. After saving, use **Recommendations** to (1) retrieve a candidate pool from the vector store and (2) rank it, showing only the **top X** results, **Add job** to parse a URL or paste and save to **My jobs**, **Resume improvement** for a chosen role, and **Interview preparation** for a structured prep plan.

You can also use **Track applications** (`/applications`) to see (and jump back into) your resume-improvement and interview-prep sessions. Sessions are stored per job/stage with a compact memory object and only the last N chat turns.

## ML experiments (local MLflow)

This repo includes a **ranking baseline** with:

- **Label**: weak supervision from cosine similarity between *job-summary* and *resume-summary* embeddings (binned to 0, 0.5, 1). The label is not derived from any single feature, so tree/linear models do not see the raw embedding.
- **Two dataset formats** per version (so you can use the right one per model):
  - **Similarity dataset** (`mock_similarity_vN.csv`): scalar features only — for **Logistic Regression, XGBoost**, etc.
  - **Embeddings dataset** (`mock_embeddings_vN.csv`): raw job and resume embedding dimensions + label — for **neural networks**.
- **Naming**: files are prefixed with `mock_` so that later you can add real user data under different names (e.g. `real_similarity_v1.csv`) and choose by name.

### Features (similarity dataset)

- `title_similarity`, `skill_overlap_count`, `location_match` (0/1), `experience_gap` (years)
- `salary_match`, `location_km` (distance in km)
- `skill_similarity`, `role_similarity`, `work_mode_similarity`, `employment_type_similarity`, `preferred_locations_similarity` (intended to be computed from LLM-extracted tags in production; mock data simulates these)

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
```

For **neural networks**, load the **embeddings** dataset (e.g. `mock_embeddings_v1.csv`) in your training script; the current CLI trains only on the similarity dataset.

On Windows PowerShell (set `PYTHONPATH` once per terminal session, then run any of the commands below):

```powershell
$env:PYTHONPATH="src"
python -m career_copilot.ml.create_ranking_dataset --n-rows 2000 --seed 7
python -m career_copilot.ml.train_logreg_mlflow --dataset-version latest
python -m career_copilot.ml.train_xgboost_mlflow --dataset-version latest --run-name xgboost-baseline
```

If you see `ModuleNotFoundError: No module named 'career_copilot'`, run `$env:PYTHONPATH="src"` in the same terminal first, or install the package in editable mode from the repo root: `pip install -e .`

Runs are stored in `data/mlflow.db` and artifacts in `data/mlflow_artifacts`. Datasets live in `data/datasets/ranking/` (`mock_similarity_vN.csv`, `mock_embeddings_vN.csv`, `manifest.json`).

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

To refresh jobs on a schedule, run `python scripts/ingestion/scheduler.py` (default: every 6 hours; edit the script to change the interval). Re-run `python scripts/rag_index/run.py` after ingestion to refresh **job embeddings** in Postgres (pgvector).

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
│   ├── ingestion/          # Job APIs (RemoteOK, Remotive, Arbeitnow, Adzuna)
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
│   ├── rag_index/
│   │   ├── Dockerfile      # RAG index image (build from repo root; see file header)
│   │   ├── requirements.txt # Pip deps for that image only
│   │   └── run.py          # Postgres jobs → embeddings in jobs_embeddings
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
