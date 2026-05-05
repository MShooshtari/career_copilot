# Routers

FastAPI route modules for the Career Copilot web app.

Each module groups HTTP endpoints for one user-facing area, keeping request handling separate from database, RAG, ML, and agent internals.

## Contents

- `home.py`, `profile.py`, and `auth.py` handle core navigation, profile setup, and authentication.
- `recommendations.py`, `jobs.py`, `my_jobs.py`, and `add_job.py` support job discovery and saved jobs.
- `resume_improvement.py` and `interview_preparation.py` expose agent-assisted workflows.
- `track_applications.py` handles application-stage tracking.
- `market_analysis.py` exposes market analysis routes.

Routers should stay thin: validate input, call the appropriate service or helper, and render or return the response.
