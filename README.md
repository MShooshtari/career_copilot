# Career Copilot

**AI Career Copilot** — A web app that helps users build a profile, get personalized job recommendations (RAG-backed), add or save jobs, improve their resume for specific roles, and prepare for interviews.

## Features

- **User profile** — Skills, experience, location, preferred roles, industries, work mode, salary range, and resume upload
- **Job recommendations** — Top jobs by vector similarity to your profile (resume + preferences) via Chroma + OpenAI embeddings
- **Job detail** — View full job description, skills, and salary
- **Add job agent** — Paste a job URL or raw text; the agent extracts title, company, location, salary, description, and skills. Optional web search (Tavily or SerpAPI) fills in missing fields. Save the result to **My jobs**.
- **My jobs** — View, edit, and manage jobs you’ve added or saved from recommendations
- **Resume improvement agent** — For a chosen job: RAG-backed chat (similar jobs + similar resumes) and one-click PDF export of the improved resume. The agent can call tools to pull extra similar jobs/resumes from the vector store when it decides more context is useful.
- **Interview preparation agent** — Structured prep plan for a specific role and company. The agent can call a web-search tool (Glassdoor, Reddit, etc.) to fetch company-specific interview insights and weave them into the guidance.
- **Track applications** — A simple tracker where each “application” is either **resume improvement** or **interview preparation** for a specific job. Each application keeps **lightweight per-job memory** (last edited resume + compact summary) and only the **last N chat turns** for continuity.

## Tech stack

- **Backend:** FastAPI, PostgreSQL (jobs, profiles, user_jobs), Chroma (vector store)
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
# Optional: run Postgres via Docker
docker compose up -d

# Create jobs table (if not already present)
psql -d career_copilot -f sql/001_create_jobs.sql

# Ingest jobs from RemoteOK, Remotive, Arbeitnow, (and Adzuna if configured)
python scripts/run_ingestion.py

# Index jobs into Chroma for RAG recommendations
python scripts/run_rag_index.py
```

### 4. Run the web app

```bash
python scripts/run_web.py
```

Then open **http://127.0.0.1:8000**. You’re redirected to `/profile`; fill in your profile and upload a resume. After saving, use **Recommendations** for jobs ranked by similarity, **Add job** to parse a URL or paste and save to **My jobs**, **Resume improvement** for a chosen role, and **Interview preparation** for a structured prep plan.

You can also use **Track applications** (`/applications`) to see (and jump back into) your resume-improvement and interview-prep sessions. Sessions are stored per job/stage with a compact memory object and only the last N chat turns.

## ML experiments (local MLflow)

This repo includes a small, **bootstrapped ranking baseline** that demonstrates:

- weak supervision labels from embedding similarity \(\{0, 0.5, 1\}\)
- feature-based Logistic Regression
- experiment tracking with **local MLflow**

### Run MLflow UI

```bash
mlflow ui --backend-store-uri "sqlite:///./data/mlflow.db" --default-artifact-root "file:./data/mlflow_artifacts"
```

Open the UI at `http://127.0.0.1:5000`.

### Versioned dataset (single source of truth)

Training uses a **versioned dataset** under `data/datasets/ranking/`. Create a new version (once or when you change data), then train pointing at that version.

**Create a dataset version** (mock data with weak-supervision labels):

```bash
PYTHONPATH=src python -m career_copilot.ml.create_ranking_dataset --n-rows 2000 --seed 7
# optional: --version v1 to name the version; otherwise v1, v2, ... auto-assigned
```

**Train using a dataset version** (logs only `dataset_version` and `dataset_path` in MLflow, not the full CSV):

```bash
PYTHONPATH=src python -m career_copilot.ml.train_logreg_mlflow --dataset-version latest
# or --dataset-version v1, v2, etc.
```

On Windows PowerShell, set `$env:PYTHONPATH="src"` then run the same `python -m ...` commands.

Runs are stored in `data/mlflow.db` and artifacts in `data/mlflow_artifacts` (model, confusion matrix). The dataset itself lives in `data/datasets/ranking/` (e.g. `v1.csv`, `v2.csv`, `manifest.json`).

## Job sources (ingestion)

| Source      | API key | Notes                    |
|------------|--------|---------------------------|
| RemoteOK   | No     | ~100 jobs per run         |
| Remotive   | No     | Remote jobs               |
| Arbeitnow  | No     | Europe-focused            |
| Adzuna     | Yes    | Set `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` in `.env` for more jobs |

To refresh jobs on a schedule, run `python scripts/scheduler.py` (default: every 6 hours; edit the script to change the interval). Re-run `python scripts/run_rag_index.py` after ingestion to update Chroma.

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
│   ├── rag/                # Chroma store, embedding, user_embedding
│   ├── ingestion/          # Job APIs (RemoteOK, Remotive, Arbeitnow, Adzuna)
│   ├── agents/             # resume_improvement, interview_preparation, add_job, track_applications, application_memory
│   ├── resume_io.py        # Resume text extraction (PDF)
│   └── resume_pdf.py       # PDF generation for improved resume
├── templates/              # Jinja2 HTML (profile, recommendations, job_detail, improve_resume, add_job, my_jobs, interview_prep)
├── configs/
│   └── config.example.env  # Example .env (use project root .env)
├── tests/                  # Pytest unit tests
├── scripts/
│   ├── run_web.py          # Start uvicorn
│   ├── run_ingestion.py    # Fetch jobs → Postgres
│   ├── run_rag_index.py    # Postgres jobs → Chroma
│   ├── scheduler.py        # Optional: run ingestion on a schedule (e.g. every 6 hours)
│   ├── repair_descriptions.py
│   └── explore_embeddings.py
├── sql/                    # 001_create_jobs.sql, 002_create_user_jobs.sql (user_jobs also created by init_schema)
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
