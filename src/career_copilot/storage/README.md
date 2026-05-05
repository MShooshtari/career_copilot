# Storage

File-storage helpers for user-uploaded and generated assets.

This package currently focuses on resume storage and keeps filesystem or object-storage details out of routers and agents.

## Contents

- `resumes.py` handles resume file persistence and retrieval helpers.

Add new storage integrations here when workflows need to read or write application files outside the database.
