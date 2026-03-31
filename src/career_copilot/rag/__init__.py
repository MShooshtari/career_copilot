"""RAG (Retrieval Augmented Generation) storage for job search."""

try:
    from career_copilot.rag.pgvector_rag import index_jobs_into_pgvector
except ModuleNotFoundError:  # pragma: no cover
    # Keep package importable even if optional deps aren't installed in a given environment.
    index_jobs_into_pgvector = None  # type: ignore[assignment]

__all__ = ["index_jobs_into_pgvector"]
