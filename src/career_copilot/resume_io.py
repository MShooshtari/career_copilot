"""Extract plain text from resume files (PDF or UTF-8 text). Used for embedding and display."""
from __future__ import annotations


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


def extract_resume_text(content_bytes: bytes, filename: str | None) -> str:
    """Extract plain text from uploaded resume (PDF or UTF-8 text)."""
    if not content_bytes:
        return ""
    filename = (filename or "").lower()
    is_pdf = filename.endswith(".pdf") or content_bytes.startswith(b"%PDF")
    if is_pdf:
        text = _extract_pdf_pypdf(content_bytes)
        if not text.strip():
            text = _extract_pdf_pymupdf(content_bytes)
        return text
    try:
        return content_bytes.decode("utf-8", errors="replace")
    except Exception:
        return ""
