"""RAG (Retrieval Augmented Generation) storage for job search."""

from career_copilot.rag.chroma_store import index_jobs_into_chroma

__all__ = ["index_jobs_into_chroma"]
