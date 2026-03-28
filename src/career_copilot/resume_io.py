"""Extract plain text from resume files (PDF, DOCX, or UTF-8 text). Used for embedding and display."""

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


def _extract_docx(content_bytes: bytes) -> str:
    """Extract plain text from a Word .docx file using python-docx."""
    try:
        from io import BytesIO

        from docx import Document
        from docx.text.paragraph import Paragraph

        doc = Document(BytesIO(content_bytes))
        parts: list[str] = []

        def walk(element, seen_tcs: set | None = None) -> None:
            """Recursively extract text in document order from any XML element."""
            if seen_tcs is None:
                seen_tcs = set()
            for child in element:
                tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if tag == "p":
                    # Leaf: extract and stop — don't descend into runs
                    text = Paragraph(child, doc).text.strip()
                    if text:
                        parts.append(text)
                elif tag == "tc":
                    # Store the element itself (not id()) so the set reference keeps it
                    # alive and prevents address reuse causing false dedup hits
                    if child not in seen_tcs:
                        seen_tcs.add(child)
                        walk(child, seen_tcs)
                else:
                    # Recurse into everything else: tbl, tr, sdt, sdtContent, body, etc.
                    walk(child, seen_tcs)

        walk(doc.element.body)
        return "\n".join(parts)
    except Exception:
        return ""


def extract_resume_text(content_bytes: bytes, filename: str | None) -> str:
    """Extract plain text from uploaded resume (PDF, DOCX, or UTF-8 text)."""
    if not content_bytes:
        return ""
    filename = (filename or "").lower()
    if filename.endswith(".docx") or filename.endswith(".doc"):
        text = _extract_docx(content_bytes)
        return text
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
