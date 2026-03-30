"""RAG (Retrieval Augmented Generation) storage for job search."""

from career_copilot.rag.azure_search_jobs import index_jobs_into_azure_search

__all__ = ["index_jobs_into_azure_search"]
