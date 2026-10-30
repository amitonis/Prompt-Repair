"""Error aggregation and feedback-packet construction (Sections 8.3 and 9.3)."""

from __future__ import annotations

from typing import Any

MAX_FAILED_EXAMPLES = 5
MAX_PASSING_EXAMPLES = 2


def build_feedback_packet(
    scored_cases: list[dict[str, Any]],
    feedback_subset_size: int | None = None,
) -> dict[str, Any]:
    """Build the editor feedback packet from scored feedback-set cases.

    Each scored case must contain: id, notice, gold, raw, and a score dict as
    produced by src.evaluator.score_response. Failed examples are capped at
    MAX_FAILED_EXAMPLES, passing examples at MAX_PASSING_EXAMPLES, all chosen
    in fixed file order (the order the cases were loaded) per the frozen
    analysis plan, Section 5.
    """
    failed = [c for c in scored_cases if not c["score"]["record_correct"]]
    passed = [c for c in scored_cases if c["score"]["record_correct"]]

    error_counts: dict[str, int] = {}
    for case in failed:
        for code in case["score"]["issues"]:
            error_counts[code] = error_counts.get(code, 0) + 1

    chosen_failed = failed[:MAX_FAILED_EXAMPLES]
    chosen_passed = passed[:MAX_PASSING_EXAMPLES]

    packet: dict[str, Any] = {
        "feedback_subset_size": feedback_subset_size or len(scored_cases),
        "error_counts": dict(sorted(error_counts.items())),
        "failed_examples": [_example(c, passed=False) for c in chosen_failed],
        "passing_examples": [_example(c, passed=True) for c in chosen_passed],
    }
    return packet


def _example(case: dict[str, Any], passed: bool) -> dict[str, Any]:
    example = {
        "id": case["id"],
        "notice": case["notice"],
        "expected": case["gold"],
        "observed": case["score"]["parsed"],
    }
    if not passed:
        example["errors"] = case["score"]["issues"]
    return example
