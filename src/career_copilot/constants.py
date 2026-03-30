"""Project-wide constants for tunable but non-secret numeric parameters.

Keep "magic numbers" (top-k, batch sizes, default limits) here so behavior can be
adjusted in one place without hunting through the codebase.
"""

from __future__ import annotations

# Single-user local demo default
DEFAULT_USER_ID = 1

# Recommendations
RECOMMENDATIONS_CANDIDATE_POOL_SIZE = 100  # retrieve this many from vector store
RECOMMENDATIONS_RERANK_WINDOW_SIZE = 15  # keep this many after re-ranking
RECOMMENDATIONS_DEFAULT_PAGE_SIZE = 5
RECOMMENDATIONS_MAX_PAGE_SIZE = 50
RECOMMENDATIONS_PAGE_SIZE_OPTIONS: tuple[int, ...] = (5, 10, 15)

# RAG / Azure Search job indexing + retrieval (batch size, doc limits)
RAG_JOB_DOC_MAX_CHARS = 6_000
RAG_JOB_UPSERT_BATCH_SIZE = 50
RAG_DEFAULT_RECOMMENDATION_N_RESULTS = 100
RAG_SIMILAR_JOBS_N_RESULTS = 5
RAG_SIMILAR_RESUMES_N_RESULTS = 5

# Interview preparation chat
APPLICATION_CHAT_MAX_STORED_MESSAGES = 20
APPLICATION_MEMORY_SUMMARY_UPDATE_EVERY_N_MESSAGES = 8

# Backward-compatible aliases (older names used in routers)
INTERVIEW_PREP_MAX_STORED_MESSAGES = APPLICATION_CHAT_MAX_STORED_MESSAGES
INTERVIEW_PREP_SUMMARY_UPDATE_EVERY_N_MESSAGES = APPLICATION_MEMORY_SUMMARY_UPDATE_EVERY_N_MESSAGES

# UI text snippet limits
JOB_DESCRIPTION_SNIPPET_MAX_CHARS = 500
