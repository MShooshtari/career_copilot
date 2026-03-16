# Career Copilot

**AI Career Copilot** — A web app that helps users build a profile, get personalized job recommendations (RAG-backed), and improve their resume for specific roles.

## Features

- **User profile** — Skills, experience, location, preferred roles, industries, work mode, salary range, and resume upload
- **Job recommendations** — Top jobs by vector similarity to your profile (resume + preferences) via Chroma + OpenAI embeddings
- **Job detail** — View full job description, skills, and salary
- **Resume improvement agent** — For a chosen job: RAG-backed chat (similar jobs + similar resumes) and one-click PDF export of the improved resume. The agent can call tools to pull extra similar jobs/resumes from the vector store when it decides more context is useful.
- **Interview preparation agent** — Structured prep plan for a specific role and company. The agent can call a web-search tool (Glassdoor, Reddit, etc.) to fetch company-specific interview insights and weave them into the guidance.

## Tech stack

- **Backend:** FastAPI, PostgreSQL (jobs + profiles), Chroma (vector store)
- **Embeddings:** OpenAI text-embedding-3-large (jobs and user profiles)
- **LLM:** OpenAI chat models with tool-calling for agentic behaviour (resume improvement + interview prep)

## Prerequisites

- Python 3.11+
- PostgreSQL
- [OpenAI API key](https://platform.openai.com/api-keys) (for embeddings and resume improvement)

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment

Create a `.env` file in the project root. Example:

```bash
# Database (required)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_DB=career_copilot
POSTGRES_PASSWORD=your_password

# Or a single DSN:
# POSTGRES_DSN=postgresql://user:password@localhost:5432/career_copilot

# OpenAI (required for embeddings and resume improvement)
OPENAI_API_KEY=sk-your-openai-key

# Optional: more jobs from Adzuna (https://developer.adzuna.com/signup)
# ADZUNA_APP_ID=your_app_id
# ADZUNA_APP_KEY=your_app_key
```

### 3. Database and jobs

The web app creates the **users**, **profiles**, and **user_skills** tables on startup. The **jobs** table must exist before indexing; create it from the SQL schema and run ingestion:

```bash
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

Then open **http://127.0.0.1:8000**. You’re redirected to `/profile`; fill in your profile and upload a resume. After saving, go to **Recommendations** to see jobs ranked by similarity to your profile.

## Job sources (ingestion)

| Source      | API key | Notes                    |
|------------|--------|---------------------------|
| RemoteOK   | No     | ~100 jobs per run         |
| Remotive   | No     | Remote jobs               |
| Arbeitnow  | No     | Europe-focused            |
| Adzuna     | Yes    | Set `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` in `.env` for more jobs |

## Project structure

```
career_copilot/
├── src/career_copilot/
│   ├── web_app.py          # FastAPI app, routers
│   ├── app_config.py       # Paths, Jinja2 templates
│   ├── schemas.py          # Pydantic request models
│   ├── utils.py            # Shared helpers
│   ├── routers/            # Route handlers (home, profile, jobs, recommendations, resume_improvement)
│   ├── database/           # db, schema, profiles, jobs, deps
│   ├── rag/                # Chroma store, embedding, user_embedding
│   ├── ingestion/          # Job APIs (RemoteOK, Remotive, Arbeitnow, Adzuna)
│   ├── agents/             # Resume improvement (context, chat, PDF text)
│   ├── resume_io.py        # Resume text extraction (PDF)
│   └── resume_pdf.py       # PDF generation for improved resume
├── templates/              # Jinja2 HTML (profile, recommendations, job_detail, improve_resume)
├── tests/                  # Pytest unit tests
├── scripts/
│   ├── run_web.py          # Start uvicorn
│   ├── run_ingestion.py    # Fetch jobs → Postgres
│   ├── run_rag_index.py    # Postgres jobs → Chroma
│   └── ...
├── sql/                    # DB schema (e.g. 001_create_jobs.sql)
├── pyproject.toml          # Ruff, pytest config
├── requirements.txt
└── requirements-dev.txt   # Ruff, pip-audit
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
