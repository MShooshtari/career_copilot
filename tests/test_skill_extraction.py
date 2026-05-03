"""Tests for dynamic job skill extraction."""

from __future__ import annotations

from career_copilot.ingestion.skill_extraction import extract_skill_tags


def test_extract_skill_tags_finds_skills_from_explicit_sections() -> None:
    text = """
    Skills:
    - customer service
    - cash handling
    - inventory management
    - forklift operation
    """

    assert extract_skill_tags(text) == [
        "Customer Service",
        "Cash Handling",
        "Inventory Management",
        "Forklift Operation",
    ]


def test_extract_skill_tags_ignores_generic_job_words() -> None:
    text = "We need a technical software engineering leader for complex systems."

    assert extract_skill_tags(text) == []


def test_extract_skill_tags_uses_source_skills_before_text_skills() -> None:
    text = "Experience with patient care, CPR certification, and electronic health records."

    assert extract_skill_tags(text, source_skills=["medical terminology", "CPR"]) == [
        "Medical Terminology",
        "CPR",
        "Patient Care",
        "Electronic Health Records",
    ]


def test_extract_skill_tags_extracts_technical_skills_without_a_taxonomy() -> None:
    text = "Experience with Python, SQL, CI/CD, and Google Cloud Platform."

    assert extract_skill_tags(text) == ["Python", "SQL", "CI/CD", "Google Cloud Platform"]
