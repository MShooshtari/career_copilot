from __future__ import annotations

import os
import uuid

from azure.storage.blob import BlobServiceClient, ContentSettings


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def resume_storage_mode() -> str:
    return (_env("RESUME_STORAGE_MODE", "db") or "db").strip().lower()


def _container_name() -> str:
    return (_env("AZURE_RESUME_CONTAINER", "resumes") or "resumes").strip()


def _conn_str() -> str | None:
    v = (_env("AZURE_STORAGE_CONNECTION_STRING") or "").strip()
    return v or None


def _blob_service() -> BlobServiceClient:
    cs = _conn_str()
    if not cs:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is required for RESUME_STORAGE_MODE=blob")
    return BlobServiceClient.from_connection_string(cs)


def put_resume(*, user_id: int, filename: str | None, content: bytes) -> tuple[str, str]:
    """
    Upload resume bytes to Azure Blob Storage.
    Returns (container, blob_name).
    """
    if resume_storage_mode() != "blob":
        raise RuntimeError("put_resume called but RESUME_STORAGE_MODE != blob")

    container = _container_name()
    ext = ""
    if filename and "." in filename:
        ext = "." + filename.rsplit(".", 1)[1].lower()
        if len(ext) > 10:
            ext = ""

    blob_name = f"user/{user_id}/{uuid.uuid4().hex}{ext}"
    service = _blob_service()
    client = service.get_blob_client(container=container, blob=blob_name)
    content_type = "application/octet-stream"
    if filename and filename.lower().endswith(".pdf"):
        content_type = "application/pdf"
    elif filename and filename.lower().endswith(".docx"):
        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    client.upload_blob(
        content,
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
    )
    return (container, blob_name)


def get_resume(*, container: str, blob_name: str) -> bytes:
    if resume_storage_mode() != "blob":
        raise RuntimeError("get_resume called but RESUME_STORAGE_MODE != blob")
    service = _blob_service()
    client = service.get_blob_client(container=container, blob=blob_name)
    downloader = client.download_blob()
    return bytes(downloader.readall())

