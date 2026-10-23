"""Optimization branches and candidate selection (Sections 9.4-9.5)."""

from __future__ import annotations

from typing import Any


def selection_key(
    candidate_scores: dict[str, Any],
    original_sections: dict[str, str],
    candidate_sections: dict[str, str],
) -> tuple[int, int, int, int, int, int]:
    """Deterministic lexicographic candidate-selection rule (Section 9.5).

    Higher is better; earlier criteria dominate. Edit distance and prompt
    length enter as negatives (smaller is better).
    """
    from .evaluator import levenshtein_chars
    from .prompt_renderer import prompt_word_count, render_template

    rendered_original = render_template(original_sections)
    rendered_candidate = render_template(candidate_sections)
    edit_distance = levenshtein_chars(rendered_original, rendered_candidate)
    return (
        candidate_scores["selection_records_correct"],
        candidate_scores["selection_fields_correct"],
        candidate_scores["feedback_records_correct"],
        candidate_scores["feedback_fields_correct"],
        -edit_distance,
        -prompt_word_count(candidate_sections),
    )


def run_targeted_branch(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """At most two attempts; one constrained section patch per attempt."""
    raise NotImplementedError("Wired to the inference adapter on Day 7")


def run_rewrite_branch(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """At most two attempts; one complete rewrite per attempt."""
    raise NotImplementedError("Wired to the inference adapter on Day 7")
