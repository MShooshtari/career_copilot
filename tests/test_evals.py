from __future__ import annotations

from career_copilot.evals.datasets import load_dataset
from career_copilot.evals.metrics import score_job_extraction, score_retrieval_case
from career_copilot.evals.rubrics import build_judge_prompt, score_rubric_judgment
from career_copilot.evals.run import (
    run_add_job_extraction_eval,
    run_all,
    run_rubric_eval,
)


def test_load_packaged_eval_dataset() -> None:
    cases = load_dataset("add_job_extraction")

    assert cases
    assert {"id", "input", "expected", "actual"}.issubset(cases[0])


def test_score_job_extraction_handles_text_numeric_and_skill_fields() -> None:
    metrics = score_job_extraction(
        {
            "title": "Senior Python API Engineer",
            "salary_min": 130000,
            "skills": ["Python", "FastAPI"],
        },
        {
            "title": "Senior Python API Engineer",
            "salary_min": 130000,
            "skills": ["Python", "FastAPI", "PostgreSQL"],
        },
    )

    assert metrics["field_title"] == 1.0
    assert metrics["field_salary_min"] == 1.0
    assert 0.0 < metrics["field_skills"] < 1.0
    assert 0.0 < metrics["field_accuracy"] < 1.0


def test_score_retrieval_case_precision_and_recall_at_k() -> None:
    metrics = score_retrieval_case(
        retrieved_ids=["a", "x", "b", "c"],
        relevant_ids=["a", "b", "z"],
        k=3,
    )

    assert metrics["retrieval_precision_at_3"] == 2 / 3
    assert metrics["retrieval_recall_at_3"] == 2 / 3


def test_rubric_judgment_scoring_and_prompt() -> None:
    judgment = {
        "scores": {
            "faithfulness": 5,
            "job_relevance": 4,
            "specificity": 3,
            "actionability": 4,
            "clarity": 5,
        }
    }

    metrics = score_rubric_judgment(judgment, rubric_name="resume_improvement")
    prompt = build_judge_prompt(
        rubric_name="resume_improvement",
        case={"resume": "Python APIs", "job": "Backend Engineer"},
        output="Tailor Python API bullets to backend responsibilities.",
    )

    assert metrics["rubric_faithfulness"] == 5.0
    assert metrics["rubric_overall"] == 4.2
    assert "Return JSON" in prompt
    assert "faithfulness" in prompt


def test_eval_runners_return_aggregate_metrics() -> None:
    add_job_metrics = run_add_job_extraction_eval(dataset="add_job_extraction")
    rubric_metrics = run_rubric_eval(dataset="rubric_outputs")
    all_metrics = run_all(k=3)

    assert add_job_metrics["cases"] >= 1
    assert "field_accuracy" in add_job_metrics
    assert rubric_metrics["judged_cases"] >= 1
    assert "add_job_extraction" in all_metrics
    assert "recommendation_ranking" in all_metrics
    assert "rag_retrieval" in all_metrics
    assert "rubric_outputs" in all_metrics
