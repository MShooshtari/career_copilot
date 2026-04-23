# Azure deployment inventory (Career Copilot API)

This doc captures the **deployable component(s)** and the **minimum environment variables** needed to run the FastAPI app on Azure Container Apps.

## Deployable component(s)

- **FastAPI web app**: `src/career_copilot/web_app.py` (module path `career_copilot.web_app:app`)
  - Serves HTML (Jinja templates) + JSON endpoints used by the UI.
  - Initializes DB schema on startup via `career_copilot.database.schema.init_schema` (see notes below).

## Required environment variables

### Database (required)

Use either a DSN or discrete env vars (see `src/career_copilot/database/db.py`):

- **Preferred (Azure)**:
  - `POSTGRES_DSN`: `postgresql://USER:PASSWORD@HOST:5432/DBNAME`
  - `POSTGRES_SSLMODE`: `require`

- **Alternative**:
  - `POSTGRES_HOST`
  - `POSTGRES_PORT` (default `5432`)
  - `POSTGRES_USER`
  - `POSTGRES_PASSWORD`
  - `POSTGRES_DB`
  - Optional: `POSTGRES_SSLMODE`

### LLM / embeddings (required for core features)

- `OPENAI_API_KEY`

### Optional (feature-specific)

- `ADZUNA_APP_ID`, `ADZUNA_APP_KEY` (job ingestion enrichment)
- `TAVILY_API_KEY` or `SERPAPI_API_KEY` (add-job agent optional web search)

## Authentication (Entra External ID)

The app supports interactive browser login (OIDC) and API bearer tokens.

- `AUTH_ENABLED`: `1` to require auth (recommended in Azure). If unset/`0`, the app runs in local demo mode.
- `SESSION_SECRET_KEY`: strong random secret for server sessions.

OIDC client settings:

- `ENTRA_CLIENT_ID`
- `ENTRA_CLIENT_SECRET`
- `ENTRA_REDIRECT_URI`: e.g. `https://<your-app-domain>/auth/callback`

Recommended (most reliable) OIDC endpoints:

- `ENTRA_METADATA_URL`: paste the **OpenID Connect metadata document** URL from your External ID app registration (looks like `https://<tenant>.ciamlogin.com/<tenant-guid>/v2.0/.well-known/openid-configuration`)
- `ENTRA_REDIRECT_URI`: e.g. `https://<your-app-domain>/auth/callback`

Optional overrides (if your tenant uses non-default endpoints):

- `ENTRA_AUTHORITY`: authority base URL (alternative to `ENTRA_METADATA_URL`; if used, include the tenant path, e.g. `https://<tenant>.ciamlogin.com/<tenant-guid>`)
- `ENTRA_TENANT_DOMAIN`: used only when `ENTRA_METADATA_URL` is not set; set to `https://<tenant>.ciamlogin.com` (or `<tenant>.ciamlogin.com`)
- `ENTRA_TENANT_ID`: used only when `ENTRA_METADATA_URL` is not set; the tenant GUID

## Azure resource checklist (MVP)

- Azure Container Apps environment
- Azure Container Registry (ACR)
- Azure Database for PostgreSQL Flexible Server
  - Ensure `pgvector` is available and your DB role can run `CREATE EXTENSION vector` (required by `init_schema`).
- Azure Key Vault (store secrets, grant managed identity access)
- Application Insights / Log Analytics (recommended)

## Notes / gotchas

- **Schema init**: `web_app.py` currently calls `init_schema` on startup. In production you may want explicit migrations (Alembic) instead of app-startup DDL.
- **Templates**: the container image must include `templates/` for the HTML UI.

