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

Goal: **ingest jobs from RemoteOK → normalize → upsert into Postgres**.

### Prereqs

- Python 3.11+
- A running PostgreSQL instance (local install is fine)

### Setup

1) Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2) Create a `.env` file in the repo root (it’s gitignored). You can copy `configs/config.example.env`, or use discrete settings:

```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_DB=career_copilot
POSTGRES_PASSWORD=your_password_here
```

3) Run ingestion:

```bash
python scripts/run_ingestion.py
```

You should see output like:

- fetched N RemoteOK rows
- upserted N jobs
- RemoteOK job count in DB: N

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
