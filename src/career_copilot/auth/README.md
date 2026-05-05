# Auth

Authentication and current-user resolution for the FastAPI app.

This package contains the route dependencies and helpers that identify the active user, support guest sessions, and integrate with Microsoft Entra ID when configured.

## Contents

- `config.py` loads authentication-related settings.
- `current_user.py` resolves the current application user.
- `deps.py` exposes authentication dependencies for routers.
- `entra.py` contains Entra ID validation helpers.
- `guest.py` supports guest-user flows.

Keep authentication boundary logic here so routers can depend on a consistent user object without duplicating identity checks.
