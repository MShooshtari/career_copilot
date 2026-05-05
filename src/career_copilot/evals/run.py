from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from career_copilot.evals.datasets import load_dataset
from career_copilot.evals.metrics import (
    aggregate_metric_dicts,
    score_job_extraction,
    score_ranking_cases,
    score_retrieval_case,
)
from career_copilot.evals.rubrics import build_judge_prompt, score_rubric_judgment


def _print_result(name: str, metrics: dict[str, float]) -> None:
    print(f"\n{name}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def run_add_job_extraction_eval(
    *,
    dataset: str,
    live: bool = False,
    extractor: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, float]:
    """
    Score add-job extraction cases.

    By default this scores saved predictions in the dataset. Set live=True to call
    the current LLM-backed extractor for each input text.
    """
    cases = load_dataset(dataset)
    if live and extractor is None:
        from career_copilot.agents.add_job import extract_job_from_text

        extractor = extract_job_from_text

    rows: list[dict[str, float]] = []
    for case in cases:
        expected = case.get("expected") or {}
        if live:
            text = str(case.get("input", {}).get("text") or "")
            actual = extractor(text) if extractor else {}
        else:
            actual = case.get("actual") or case.get("prediction") or {}
        rows.append(score_job_extraction(actual, expected))
    return {"cases": float(len(cases)), **aggregate_metric_dicts(rows)}


def run_recommendation_ranking_eval(*, dataset: str, k: int) -> dict[str, float]:
    cases = load_dataset(dataset)
    return {"cases": float(len(cases)), **score_ranking_cases(cases, k=k)}


def run_rag_retrieval_eval(*, dataset: str, k: int) -> dict[str, float]:
    cases = load_dataset(dataset)
    rows = [
        score_retrieval_case(
            case.get("retrieved_ids", []),
            case.get("relevant_ids", []),
            k=k,
        )
        for case in cases
    ]
    return {"cases": float(len(cases)), **aggregate_metric_dicts(rows)}


def run_rubric_eval(*, dataset: str) -> dict[str, float]:
    """
    Score already-judged rubric cases.

    Cases without judgment are counted so teams can see pending human/judge-model work.
    """
    cases = load_dataset(dataset)
    rows: list[dict[str, float]] = []
    pending = 0
    for case in cases:
        rubric_name = str(case.get("rubric") or "")
        judgment = case.get("judgment")
        if not rubric_name or not judgment:
            pending += 1
            continue
        rows.append(score_rubric_judgment(judgment, rubric_name=rubric_name))
    return {
        "cases": float(len(cases)),
        "judged_cases": float(len(rows)),
        "pending_cases": float(pending),
        **aggregate_metric_dicts(rows),
    }


def print_judge_prompts(*, dataset: str) -> None:
    """Print judge prompts for rubric cases that do not yet have judgments."""
    for case in load_dataset(dataset):
        rubric_name = str(case.get("rubric") or "")
        if not rubric_name or case.get("judgment"):
            continue
        print("\n" + "=" * 80)
        print(
            build_judge_prompt(
                rubric_name=rubric_name, case=case, output=str(case.get("output") or "")
            )
        )


def run_all(*, live_add_job: bool = False, k: int = 3) -> dict[str, dict[str, float]]:
    return {
        "add_job_extraction": run_add_job_extraction_eval(
            dataset="add_job_extraction",
            live=live_add_job,
        ),
        "recommendation_ranking": run_recommendation_ranking_eval(
            dataset="recommendation_ranking",
            k=k,
        ),
        "rag_retrieval": run_rag_retrieval_eval(dataset="rag_retrieval", k=k),
        "rubric_outputs": run_rubric_eval(dataset="rubric_outputs"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Career Copilot AI evaluation benchmarks.")
    parser.add_argument(
        "--suite",
        choices=["all", "add-job", "ranking", "rag", "rubric", "judge-prompts"],
        default="all",
    )
    parser.add_argument("--dataset", help="Dataset name or JSONL path. Defaults to suite example.")
    parser.add_argument("--k", type=int, default=3, help="Top-k for ranking and retrieval metrics.")
    parser.add_argument(
        "--live-add-job",
        action="store_true",
        help="Call the current add-job LLM extractor instead of scoring saved predictions.",
    )
    args = parser.parse_args()

    if args.suite == "all":
        for name, metrics in run_all(live_add_job=args.live_add_job, k=args.k).items():
            _print_result(name, metrics)
    elif args.suite == "add-job":
        _print_result(
            "add_job_extraction",
            run_add_job_extraction_eval(
                dataset=args.dataset or "add_job_extraction",
                live=args.live_add_job,
            ),
        )
    elif args.suite == "ranking":
        _print_result(
            "recommendation_ranking",
            run_recommendation_ranking_eval(
                dataset=args.dataset or "recommendation_ranking",
                k=args.k,
            ),
        )
    elif args.suite == "rag":
        _print_result(
            "rag_retrieval",
            run_rag_retrieval_eval(dataset=args.dataset or "rag_retrieval", k=args.k),
        )
    elif args.suite == "rubric":
        _print_result("rubric_outputs", run_rubric_eval(dataset=args.dataset or "rubric_outputs"))
    elif args.suite == "judge-prompts":
        print_judge_prompts(dataset=args.dataset or "rubric_outputs")


if __name__ == "__main__":
    main()
