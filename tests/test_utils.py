"""Tests for career_copilot.utils."""

from __future__ import annotations

from career_copilot.utils import strip_nul


def test_strip_nul_empty() -> None:
    assert strip_nul("") == ""


def test_strip_nul_no_nul() -> None:
    assert strip_nul("hello world") == "hello world"


def test_strip_nul_single_nul() -> None:
    assert strip_nul("hello\x00world") == "helloworld"


def test_strip_nul_multiple_nuls() -> None:
    assert strip_nul("a\x00b\x00c") == "abc"


def test_strip_nul_only_nuls() -> None:
    assert strip_nul("\x00\x00\x00") == ""
