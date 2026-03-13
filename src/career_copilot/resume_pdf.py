"""
Generate a simple PDF from resume text (improved-resume download).

Layout: A4, tighter margins so content uses most of the page width/height.
"""
from __future__ import annotations

import re
from io import BytesIO

# A4 595×842 pt; ~0.7" side margins (50pt) → content width ~495pt → ~85 chars at 11pt
PDF_RESUME_MAX_LINE_CHARS = 85
PDF_LINE_HEIGHT = 13.0
PDF_HEADER_LINE_HEIGHT = 16.0
PDF_PAGE_TOP = 50.0
PDF_PAGE_BOTTOM = 792.0  # use most of the page height
PDF_CONTENT_LEFT = 50.0

PDF_SECTION_HEADERS = frozenset({
    "summary", "experience", "work experience", "education", "skills",
    "technical skills", "core competencies", "projects", "contact", "objective",
})


def _wrap_line(line: str, max_chars: int = PDF_RESUME_MAX_LINE_CHARS) -> list[str]:
    """Wrap a single line to max_chars, breaking at word boundaries."""
    if len(line) <= max_chars:
        return [line] if line.strip() else []
    out: list[str] = []
    rest = line
    while rest:
        rest = rest.lstrip()
        if not rest:
            break
        if len(rest) <= max_chars:
            out.append(rest)
            break
        chunk = rest[: max_chars + 1]
        last_space = chunk.rfind(" ")
        if last_space <= 0:
            last_space = max_chars
        out.append(rest[:last_space].rstrip())
        rest = rest[last_space:]
    return out


def clean_resume_text_for_pdf(text: str) -> str:
    """
    Normalize LLM output for PDF: strip code fences, remove **bold**, collapse blank lines.
    """
    if not text:
        return ""
    s = text.strip()
    lines = s.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    s = "\n".join(lines)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def build_resume_pdf(resume_text: str) -> bytes:
    """
    Render resume text into a simple A4 PDF with word wrap and section-header styling.
    Uses built-in Helvetica only. Returns PDF bytes.
    """
    text = clean_resume_text_for_pdf(resume_text or "")
    if not text:
        text = (
            "Updated resume could not be generated automatically.\n\n"
            "Summary:\n"
            "  (Add a brief summary of your experience and target role here.)\n\n"
            "Experience:\n"
            "  (List your roles, responsibilities, and impact here.)\n\n"
            "Skills:\n"
            "  (List your key skills here.)\n"
        )

    import pymupdf

    buffer = BytesIO()
    doc = pymupdf.open()
    page = doc.new_page()
    y = PDF_PAGE_TOP
    lines = text.splitlines() or [" "]
    for raw_line in lines:
        line = raw_line.rstrip() or " "
        wrapped = _wrap_line(line, PDF_RESUME_MAX_LINE_CHARS)
        if not wrapped:
            y += PDF_LINE_HEIGHT * 0.5
            continue
        is_section_header = (
            len(wrapped) == 1
            and len(line) < 50
            and line.lower().rstrip(":") in PDF_SECTION_HEADERS
        )
        for part in wrapped:
            if y > PDF_PAGE_BOTTOM:
                page = doc.new_page()
                y = PDF_PAGE_TOP
            fontsize = 12 if is_section_header else 11
            page.insert_text((PDF_CONTENT_LEFT, y), part, fontsize=fontsize, fontname="helv")
            y += PDF_HEADER_LINE_HEIGHT if is_section_header else PDF_LINE_HEIGHT
        if is_section_header:
            y += 2.0

    doc.save(buffer)
    doc.close()
    buffer.seek(0)
    return buffer.read()
