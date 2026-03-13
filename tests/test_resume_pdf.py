"""Tests for career_copilot.resume_pdf."""
from __future__ import annotations

import importlib.util

import pytest

from career_copilot.resume_pdf import (
    PDF_RESUME_MAX_LINE_CHARS,
    build_resume_pdf,
    clean_resume_text_for_pdf,
    _wrap_line,
)


def test_clean_resume_text_empty() -> None:
    assert clean_resume_text_for_pdf("") == ""
    assert clean_resume_text_for_pdf("   ") == ""


def test_clean_resume_text_strips_code_fences() -> None:
    text = "```\nSummary here\n```"
    out = clean_resume_text_for_pdf(text)
    assert "```" not in out
    assert "Summary here" in out


def test_clean_resume_text_removes_bold_markers() -> None:
    assert "**" not in clean_resume_text_for_pdf("**Bold** text")
    assert "Bold" in clean_resume_text_for_pdf("**Bold** text")


def test_clean_resume_text_collapses_blank_lines() -> None:
    text = "Line 1\n\n\n\nLine 2"
    out = clean_resume_text_for_pdf(text)
    assert "Line 1" in out and "Line 2" in out
    assert "\n\n\n" not in out


def test_wrap_line_short() -> None:
    short = "hello"
    assert _wrap_line(short) == [short]


def test_wrap_line_empty() -> None:
    assert _wrap_line("") == []
    assert _wrap_line("   ") == []


def test_wrap_line_long_breaks_at_word() -> None:
    long_line = "a " * (PDF_RESUME_MAX_LINE_CHARS // 2 + 1)
    out = _wrap_line(long_line, max_chars=20)
    assert len(out) >= 2
    for chunk in out:
        assert len(chunk) <= 21


@pytest.mark.skipif(
    importlib.util.find_spec("pymupdf") is None,
    reason="pymupdf not installed",
)
def test_build_resume_pdf_returns_bytes() -> None:
    pdf = build_resume_pdf("Name\nSummary line.")
    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF")

