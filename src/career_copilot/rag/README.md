# RAG

Retrieval-augmented generation utilities for job and profile matching.

This package handles text chunking, embeddings, pgvector retrieval, and document-shaping helpers used by recommendations and agent workflows.

## Contents

- `embedding.py` wraps embedding generation and vector settings.
- `chunk_text.py` splits source text into retrievable chunks.
- `job_document.py` and `job_chunks.py` shape job records for retrieval.
- `pgvector_rag.py` queries PostgreSQL vector indexes.
- `user_embedding.py` builds user-profile embeddings.

Keep retrieval concerns here so agents and routers can consume search results without knowing the vector-store details.
