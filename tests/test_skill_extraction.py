"""Tests for dynamic job skill extraction."""

from __future__ import annotations

from types import SimpleNamespace

from career_copilot.ingestion.skill_extraction import (
    extract_ai_resume_skill_tags,
    extract_ai_skill_tags,
    extract_skill_tags,
    skill_specificity_score,
)


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content
        self.last_kwargs = {}

    def create(self, **_kwargs):
        self.last_kwargs = _kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


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


def test_extract_skill_tags_skips_generic_source_tags_and_keeps_specific_text_tags() -> None:
    text = "Experience with Python, FastAPI, LangChain, and vector databases."

    assert extract_skill_tags(
        text,
        source_skills=[
            "engineering",
            "engineer",
            "software",
            "digital nomad",
            "code",
            "building",
        ],
    ) == ["Python", "FastAPI", "LangChain", "Vector Databases"]


def test_extract_skill_tags_dedupes_case_variants_after_filtering() -> None:
    text = "Skills: python, Python, PYTHON, SQL."

    assert extract_skill_tags(text) == ["Python", "SQL"]


def test_extract_ai_skill_tags_normalizes_and_filters_model_output() -> None:
    client = _FakeClient(
        '{"skills": ["python", "engineering", "CPR certification", "team player", '
        '"interviewing", "increase your earnings with higher rates", "SQL"]}'
    )

    assert extract_ai_skill_tags("Job description", client=client) == [
        "Python",
        "CPR",
        "SQL",
    ]


def test_extract_ai_skill_tags_drops_vague_skills_when_specific_skills_exist() -> None:
    client = _FakeClient(
        '{"skills": ["problem solving", "attention to detail", "Python", "Vector databases"]}'
    )

    assert extract_ai_skill_tags("Job description", client=client) == [
        "Python",
        "Vector Databases",
    ]


def test_extract_ai_skill_tags_keeps_vague_skills_when_no_specific_skills_exist() -> None:
    client = _FakeClient('{"skills": ["problem solving", "attention to detail"]}')

    assert extract_ai_skill_tags("Job description", client=client) == [
        "Problem Solving",
        "Attention to Detail",
    ]


def test_skill_specificity_score_demotes_vague_soft_skills() -> None:
    assert skill_specificity_score("Problem Solving") < skill_specificity_score("Python")
    assert skill_specificity_score("Attention to Detail") < skill_specificity_score(
        "Vector Databases"
    )
    assert skill_specificity_score("Interviewing") == 0.0


def test_extract_ai_skill_tags_returns_empty_for_invalid_json() -> None:
    client = _FakeClient("not json")

    assert extract_ai_skill_tags("Job description", client=client) == []


def test_extract_ai_resume_skill_tags_uses_resume_prompt_and_normalization() -> None:
    client = _FakeClient('{"skills": ["python", "AWS", "team player", "communication"]}')

    assert extract_ai_resume_skill_tags("Resume text", client=client) == [
        "Python",
        "AWS",
    ]
    messages = client.completions.last_kwargs["messages"]
    assert "resumes" in messages[0]["content"]
    assert "Resume:" in messages[1]["content"]
