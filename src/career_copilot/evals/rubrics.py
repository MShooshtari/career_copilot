from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RubricCriterion:
    name: str
    description: str
    min_score: int = 1
    max_score: int = 5


@dataclass(frozen=True)
class Rubric:
    name: str
    criteria: tuple[RubricCriterion, ...]


RESUME_IMPROVEMENT_RUBRIC = Rubric(
    name="resume_improvement",
    criteria=(
        RubricCriterion(
            "faithfulness",
            "Does not invent employers, degrees, certifications, years of experience, or skills.",
        ),
        RubricCriterion(
            "job_relevance",
            "Targets the role, company context, responsibilities, and required skills.",
        ),
        RubricCriterion(
            "specificity",
            "Provides concrete rewrites or advice rather than generic resume tips.",
        ),
        RubricCriterion(
            "actionability",
            "Gives the user clear next edits they can apply directly.",
        ),
        RubricCriterion(
            "clarity",
            "Is concise, organized, and easy to follow.",
        ),
    ),
)

INTERVIEW_PREPARATION_RUBRIC = Rubric(
    name="interview_preparation",
    criteria=(
        RubricCriterion(
            "groundedness",
            "Separates known job/resume facts from web-search context or general advice.",
        ),
        RubricCriterion(
            "role_relevance",
            "Prepares for the stated interview type and the target role.",
        ),
        RubricCriterion(
            "company_specificity",
            "Uses company-specific context when available without overclaiming.",
        ),
        RubricCriterion(
            "practice_quality",
            "Includes realistic questions, STAR prompts, or drills the user can practice.",
        ),
        RubricCriterion(
            "clarity",
            "Is concise, organized, and easy to follow.",
        ),
    ),
)

APPLICATION_TRACKING_RUBRIC = Rubric(
    name="application_tracking",
    criteria=(
        RubricCriterion(
            "state_awareness",
            "Uses the user's application/job context and does not confuse jobs or stages.",
        ),
        RubricCriterion(
            "next_step_quality",
            "Suggests useful next actions for the current application stage.",
        ),
        RubricCriterion(
            "memory_faithfulness",
            "Does not introduce facts that are absent from saved application memory.",
        ),
        RubricCriterion(
            "clarity",
            "Is concise, organized, and easy to follow.",
        ),
    ),
)

RUBRICS = {
    rubric.name: rubric
    for rubric in (
        RESUME_IMPROVEMENT_RUBRIC,
        INTERVIEW_PREPARATION_RUBRIC,
        APPLICATION_TRACKING_RUBRIC,
    )
}


def build_judge_prompt(*, rubric_name: str, case: dict[str, Any], output: str) -> str:
    """
    Build a framework-neutral judge prompt.

    This can be sent to a judge model directly today, or moved into LangSmith later.
    """
    rubric = RUBRICS[rubric_name]
    criteria = "\n".join(
        f"- {criterion.name} ({criterion.min_score}-{criterion.max_score}): {criterion.description}"
        for criterion in rubric.criteria
    )
    return (
        f"You are evaluating the Career Copilot {rubric.name} AI output.\n\n"
        "Score each criterion as an integer from 1 to 5 and explain any score below 4. "
        "Penalize unsupported claims, invented user credentials, and vague advice.\n\n"
        f"Criteria:\n{criteria}\n\n"
        f"Case:\n{case}\n\n"
        f"Output:\n{output}\n\n"
        "Return JSON with keys: scores, overall_score, failures, explanation."
    )


def score_rubric_judgment(judgment: dict[str, Any], *, rubric_name: str) -> dict[str, float]:
    """Normalize a human or judge-model rubric judgment into numeric metrics."""
    rubric = RUBRICS[rubric_name]
    raw_scores = judgment.get("scores") if isinstance(judgment, dict) else {}
    scores = raw_scores if isinstance(raw_scores, dict) else {}
    out: dict[str, float] = {}
    values: list[float] = []
    for criterion in rubric.criteria:
        raw_value = scores.get(criterion.name)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = 0.0
        out[f"rubric_{criterion.name}"] = value
        values.append(value)
    out["rubric_overall"] = sum(values) / len(values) if values else 0.0
    return out
