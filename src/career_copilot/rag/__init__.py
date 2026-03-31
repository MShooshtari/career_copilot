"""RAG (Retrieval Augmented Generation) storage for job search."""

from career_copilot.rag.pgvector_rag import index_jobs_into_pgvector

__all__ = ["index_jobs_into_pgvector"]
