"""Metrics and report generation (Sections 11 and 9A.6)."""

from __future__ import annotations

from typing import Any


def restoration_ratio(defective_accuracy: float, repaired_accuracy: float, oracle_accuracy: float) -> float | None:
    """(repaired - defective) / (oracle - defective); None when the denominator is zero."""
    denominator = oracle_accuracy - defective_accuracy
    if denominator == 0:
        return None
    return (repaired_accuracy - defective_accuracy) / denominator


def recovery_regression(
    original_correct: list[bool],
    repaired_correct: list[bool],
) -> dict[str, Any]:
    """Paired recovery and regression counts on the same notices (Section 11.4)."""
    if len(original_correct) != len(repaired_correct):
        raise ValueError("Outcome lists must be paired case by case")
    original_failures = sum(1 for o in original_correct if not o)
    original_successes = sum(1 for o in original_correct if o)
    recovered = sum(1 for o, r in zip(original_correct, repaired_correct) if not o and r)
    regressed = sum(1 for o, r in zip(original_correct, repaired_correct) if o and not r)
    return {
        "recovered": recovered,
        "regressed": regressed,
        "recovery_rate": recovered / original_failures if original_failures else None,
        "regression_rate": regressed / original_successes if original_successes else None,
    }


def build_report(*args: Any, **kwargs: Any) -> Any:
    """Populate the blank results table (Section 11.7) from saved run records."""
    raise NotImplementedError("Runs after final-test sweeps (Day 11-12)")
