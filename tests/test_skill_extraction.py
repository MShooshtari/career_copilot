"""Tests for deterministic job skill extraction."""

from __future__ import annotations

from career_copilot.ingestion.skill_extraction import extract_skill_tags


def test_extract_skill_tags_finds_specific_multiword_skills() -> None:
    text = """
    You will design A/B testing experiments, build CI/CD pipelines, and use
    PostgreSQL with dbt models for product analytics.
    """

    assert extract_skill_tags(text) == [
        "A/B Testing",
        "CI/CD",
        "PostgreSQL",
        "dbt",
        "Product Analytics",
    ]


def test_extract_skill_tags_ignores_generic_job_words() -> None:
    text = "We need a technical software engineering leader for complex systems."

    assert extract_skill_tags(text) == []


def test_extract_skill_tags_canonicalizes_aliases() -> None:
    text = "Experience with continuous integration, K8s, Google Cloud Platform, and PySpark."

    assert extract_skill_tags(text) == ["CI/CD", "Kubernetes", "GCP", "Apache Spark"]


def test_extract_skill_tags_does_not_match_skill_inside_longer_word() -> None:
    assert extract_skill_tags("The role focuses on javascript-heavy frontend work.") == [
        "JavaScript"
    ]
    assert "Java" not in extract_skill_tags("javascript-heavy frontend work")
