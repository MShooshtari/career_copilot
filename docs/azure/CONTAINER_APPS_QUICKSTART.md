# Azure Container Apps quickstart (API)

This is a **manual MVP** path using `az` CLI.

## Prereqs

- Azure subscription
- `az` CLI logged in
- A resource group name and region (example uses `eastus`)

## 1) Create resource group

```bash
az group create -n career-copilot-rg -l eastus
```

## 2) Create ACR (container registry)

```bash
az acr create -g career-copilot-rg -n careercopilotacr --sku Basic
az acr login -n careercopilotacr
```

## 3) Build and push the API image

From repo root:

```bash
az acr build -r careercopilotacr -t career-copilot-api:latest .
```

## 4) Create Container Apps environment

```bash
az extension add --name containerapp --upgrade

az containerapp env create \
  -g career-copilot-rg \
  -n career-copilot-aca-env \
  -l eastus
```

## 5) Create Azure Postgres Flexible Server

Create a server + database, then set:

- `POSTGRES_DSN` (recommended) and `POSTGRES_SSLMODE=require`
- Ensure `pgvector` is available and your DB role can run `CREATE EXTENSION vector` (required by `career_copilot.database.schema.init_schema`).

## 6) Create Key Vault + secrets (recommended)

Store secrets like `POSTGRES_DSN`, `OPENAI_API_KEY`, and Entra External ID credentials in Key Vault, then reference them as Container Apps secrets/env vars.

## 7) Deploy the API Container App

Example (replace values):

```bash
az containerapp create \
  -g career-copilot-rg \
  -n career-copilot-api \
  --environment career-copilot-aca-env \
  --image careercopilotacr.azurecr.io/career-copilot-api:latest \
  --ingress external \
  --target-port 8000 \
  --registry-server careercopilotacr.azurecr.io \
  --env-vars \
    PORT=8000 \
    POSTGRES_DSN="..." \
    POSTGRES_SSLMODE=require \
    OPENAI_API_KEY="..." \
    AUTH_ENABLED=1 \
    SESSION_SECRET_KEY="..." \
    ENTRA_CLIENT_ID="..." \
    ENTRA_CLIENT_SECRET="..." \
    ENTRA_TENANT_DOMAIN="..." \
    ENTRA_REDIRECT_URI="https://<your-app-domain>/auth/callback"
```

## 8) Verify

- Check logs:

```bash
az containerapp logs show -g career-copilot-rg -n career-copilot-api --follow
```

- Hit `/healthz` (added in the auth step) and load `/profile`.

