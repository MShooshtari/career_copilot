# Career Copilot

**AI Career Copilot** — A web app that helps people discover jobs, prepare for interviews, and get personalized recommendations. Built with RAG, recommendation systems, and user modeling.

---

## Idea

Career Copilot helps users:

- **Paste resume** — Ingest and analyze resume content
- **Connect LinkedIn** — Optional profile integration
- **Track job applications** — Application status and history
- **Get interview prep** — Tailored preparation for specific roles
- **Get job recommendations** — Personalized job suggestions

---

## Features

## Local quickstart (Phase 1)

Goal: **ingest jobs from multiple APIs → normalize → upsert into Postgres** (no duplicates; re-runs upsert by `source` + `source_id`).

### Job sources (ingestion)

| Source | API key | Notes |
|--------|--------|--------|
| **RemoteOK** | None | ~100 jobs per run |
| **Remotive** | None | Remote jobs; rate limit ~4 req/day |
| **Arbeitnow** | None | Europe, ATS-backed; large feed |
| **Adzuna** | **Yes** (free) | Get [App ID & App Key](https://developer.adzuna.com/signup); add to `.env` as `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` for 100s more jobs (GB/US, paginated) |

### Prereqs

- Python 3.11+
- A running PostgreSQL instance (local install is fine)

### Setup

1) Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2) Create a `.env` file in the repo root (copy from `.env.example`). Required: DB and OpenAI (for embeddings); optional: Adzuna:

```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_DB=career_copilot
POSTGRES_PASSWORD=your_password_here

# Required for embeddings (job + user profile RAG). Get key: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-your-openai-key-here

# Optional: more jobs from Adzuna
# ADZUNA_APP_ID=your_app_id
# ADZUNA_APP_KEY=your_app_key
```

3) Run ingestion:

```bash
python scripts/run_ingestion.py
```

You should see per-source fetch counts and total upserted, e.g.:

- `remoteok: 100 fetched | remotive: N fetched | arbeitnow: N fetched | adzuna: N fetched`
- `Upserted N | Total in DB: N (by source: {...})`

### DB schema

The Phase 1 schema lives in `sql/001_create_jobs.sql` and is applied automatically by `scripts/run_ingestion.py`.

### Data ingestion

Job postings are collected from multiple sources:

| Source | Type |
|--------|------|
| [Adzuna API](https://developer.adzuna.com/) | API |
| [Remotive API](https://remotive.com/api) | API |
| Greenhouse boards | API |
| Lever boards | API |
| LinkedIn | Optional scraping |

**Pipeline:**

```
API ingestion → Kafka → Raw storage (Blob) → Feature pipeline → DB
```

**Stored fields:** job title, company, description, skills, location, salary.

---

### RAG (Retrieval-augmented generation)

Users can ask questions like:

- *"How should I prepare for this role?"*

The system retrieves:

- Job description
- Similar jobs
- Interview questions
- Resume suggestions

**RAG pipeline:**

```
User query
   ↓
Embedding
   ↓
Vector search (Azure AI Search)
   ↓
Relevant docs
   ↓
LLM answer
```

---

### Recommendation system

**Two-stage recommender:**

#### Stage 1 — Candidate retrieval

- Vector similarity: `resume_embedding` vs `job_embedding`
- Retrieve **top 500** jobs

#### Stage 2 — Ranking model

- **Features:**  
  `user_skill_match`, `location_match`, `salary_match`, `company_popularity`, `recency`, `similar_users_click_rate`
- **Models:** LightGBM, XGBoost, or neural ranker
- **Output:** Top 20 jobs

---

### Cold start

| Case | Approach |
|------|----------|
| **New user** | Onboarding: skills, experience, location, desired salary → generate initial embedding |
| **New jobs** | Content embeddings instead of collaborative filtering |

---

### Login vs anonymous users

| Mode | Data used | Quality |
|------|-----------|--------|
| **Logged in** | User profile, interaction history, clicks, applications | Better recommendations |
| **Anonymous** | Session (cookies, session embedding) | Less accurate but functional |

---

### MCP usage

MCP tools exposed to the LLM:

- `search_jobs()`
- `recommend_jobs()`
- `fetch_company_info()`
- `schedule_interview_prep()`

**Example:**  
User: *"Find remote ML engineer jobs in Canada."*  
→ LLM calls MCP → `search_jobs`.

---

### Data collection

Events tracked for training and analytics:

- Clicks
- Job saves
- Job applications
- Resume edits
- Time spent

**Storage:** Event stream (for later model training).

---

## Architecture overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Job APIs    │────▶│   Kafka     │────▶│ Blob (raw)  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                                │
                                                ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ User query  │────▶│  Embedding  │────▶│ Azure AI    │
│ / Resume    │     │  + Vector   │     │ Search      │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                                │
                                                ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Top 20 jobs │◀────│  Ranker    │◀────│ Top 500     │
│ + RAG answer│     │ (LightGBM) │     │ candidates  │
└─────────────┘     └─────────────┘     └─────────────┘
```

---

## License

See [LICENSE](LICENSE) (if applicable).
