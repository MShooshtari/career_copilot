"""Extract plain text from resume files (PDF or UTF-8 text). Used for embedding and display."""

from __future__ import annotations

import json
import os
import tempfile

from career_copilot.mcp_client import McpStdioClient, McpStdioServerSpec


def _extract_pdf_pypdf(content_bytes: bytes) -> str:
    """Extract text using pypdf. Returns empty string on failure or no text."""
    try:
        from io import BytesIO

        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content_bytes))
        parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
        return "\n\n".join(parts) if parts else ""
    except Exception:
        return ""


def _extract_pdf_pymupdf(content_bytes: bytes) -> str:
    """Extract text using PyMuPDF (often better for real-world PDFs). Returns empty on failure."""
    try:
        import pymupdf

        doc = pymupdf.open(stream=content_bytes, filetype="pdf")
        parts = [page.get_text() for page in doc]
        doc.close()
        text = "\n\n".join(p for p in parts if p.strip())
        return text
    except Exception:
        return ""


def _docling_mcp_spec() -> McpStdioServerSpec:
    """
    Configure how to launch a Docling MCP server subprocess.

    Defaults to `python -m docling_mcp.servers.mcp_server --transport stdio`.

    Env vars:
      - CAREER_COPILOT_DOCLING_MCP_COMMAND
      - CAREER_COPILOT_DOCLING_MCP_ARGS (JSON array of strings)
    """
    command = (os.environ.get("CAREER_COPILOT_DOCLING_MCP_COMMAND") or "python").strip()
    args_raw = os.environ.get("CAREER_COPILOT_DOCLING_MCP_ARGS")
    if args_raw:
        args = json.loads(args_raw)
        if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
            raise ValueError("CAREER_COPILOT_DOCLING_MCP_ARGS must be a JSON array of strings")
    else:
        args = ["-m", "docling_mcp.servers.mcp_server", "--transport", "stdio"]
    return McpStdioServerSpec(command=command, args=args)


def _extract_pdf_docling_mcp(content_bytes: bytes) -> str:
    """
    Extract a layout-aware markdown representation using Docling MCP, then return it as text.

    This tends to preserve structure (headings, lists) better than raw PDF text extraction.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="career-copilot-docling-") as td:
            path = os.path.join(td, "resume.pdf")
            with open(path, "wb") as f:
                f.write(content_bytes)

            client = McpStdioClient(_docling_mcp_spec())
            try:
                conv = client.call_tool("convert_document_into_docling_document", {"source": path})
                if not isinstance(conv, dict) or "document_key" not in conv:
                    return ""
                doc_key = conv["document_key"]
                exported = client.call_tool(
                    "export_docling_document_to_markdown",
                    {"document_key": doc_key, "max_size": 200_000},
                )
                if isinstance(exported, dict) and isinstance(exported.get("markdown"), str):
                    return exported["markdown"]
                return ""
            finally:
                client.close()
    except Exception:
        # If Docling MCP is not installed or fails (e.g. protocol/transport issues),
        # fall back to builtin extractors.
        return ""


def extract_resume_text(content_bytes: bytes, filename: str | None) -> str:
    """Extract plain text from uploaded resume (PDF or UTF-8 text)."""
    if not content_bytes:
        return ""
    filename = (filename or "").lower()
    is_pdf = filename.endswith(".pdf") or content_bytes.startswith(b"%PDF")
    if is_pdf:
        mode = (os.environ.get("RESUME_PDF_INGEST") or "builtin").strip().lower()
        if mode in ("docling_mcp", "mcp", "docling"):
            text = _extract_pdf_docling_mcp(content_bytes)
            if text.strip():
                return text
        text = _extract_pdf_pypdf(content_bytes)
        if not text.strip():
            text = _extract_pdf_pymupdf(content_bytes)
        return text
    try:
        return content_bytes.decode("utf-8", errors="replace")
    except Exception:
        return ""
