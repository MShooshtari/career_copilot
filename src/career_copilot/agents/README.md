# Agents

LLM-backed workflow logic for job extraction, resume improvement, interview preparation, and application memory.

This package owns prompt orchestration, tool usage, and compact memory helpers used by the app's agent-assisted features.

## Contents

- `add_job.py` extracts structured job details from URLs or pasted text.
- `resume_improvement.py` helps tailor resumes to a selected job.
- `interview_preparation.py` creates role- and company-specific interview prep guidance.
- `track_applications.py` supports application tracking workflows.
- `application_memory.py` stores compact per-job workflow memory.

Keep agent-side prompting and tool orchestration here so routers remain focused on HTTP concerns.
