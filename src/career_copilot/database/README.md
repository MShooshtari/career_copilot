# Database

Database access helpers, schema initialization, and persistence functions.

This package owns PostgreSQL connection handling and the query functions used by routers, agents, and background workflows.

## Contents

- `db.py` creates database connections and common execution helpers.
- `schema.py` initializes application tables and indexes.
- `users.py`, `profiles.py`, `jobs.py`, and `applications.py` contain domain-specific persistence functions.
- `deps.py` exposes database dependencies for FastAPI routes.
- `test_connection.py` provides a simple connectivity check.

Prefer adding database-specific behavior here instead of embedding SQL directly in route handlers.
