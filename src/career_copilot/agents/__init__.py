"""Chatbot agents for Career Copilot (resume improvement, etc.)."""

try:
    from career_copilot.agents.resume_improvement import (
        build_resume_improvement_context,
        chat_resume_improvement,
        generate_full_resume,
        get_initial_resume_analysis,
    )
except ModuleNotFoundError:  # pragma: no cover
    # Keep package importable when optional deps aren't installed.
    build_resume_improvement_context = None  # type: ignore[assignment]
    chat_resume_improvement = None  # type: ignore[assignment]
    generate_full_resume = None  # type: ignore[assignment]
    get_initial_resume_analysis = None  # type: ignore[assignment]

__all__ = [
    "build_resume_improvement_context",
    "chat_resume_improvement",
    "generate_full_resume",
    "get_initial_resume_analysis",
]
