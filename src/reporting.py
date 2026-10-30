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


def build_report(
    repair_payload: dict[str, Any],
    final_payload: dict[str, Any],
    *,
    oracle_accuracy: float | None = None,
    defect_accuracy: float | None = None,
) -> dict[str, Any]:
    """Build a JSON-compatible report from a repair sweep and final test.

    The function reports every locked arm, including unchanged or rejected
    repairs. Statistical comparisons remain separate because the project has
    one retained setting (D6), so cross-setting tests are not meaningful yet.
    """
    if not isinstance(repair_payload, dict) or not isinstance(final_payload, dict):
        raise TypeError("repair_payload and final_payload must be dictionaries")
    final_results = final_payload.get("results", {})
    table: list[dict[str, Any]] = []
    for arm, result in final_results.items():
        metrics = result.get("metrics", {})
        attempted = int(metrics.get("attempted", 0))
        correct = int(metrics.get("selection_records_correct", 0))
        accuracy = correct / attempted if attempted else None
        row: dict[str, Any] = {
            "arm": arm,
            "test_attempted": attempted,
            "test_strict_accuracy": accuracy,
            "test_correct": correct,
            "test_invalid": metrics.get("selection_invalid"),
            "locked_sections": result.get("sections"),
        }
        if oracle_accuracy is not None and defect_accuracy is not None and accuracy is not None:
            row["restoration_ratio"] = restoration_ratio(defect_accuracy, accuracy, oracle_accuracy)
        table.append(row)
    return {
        "study_scope": {
            "setting": repair_payload.get("setting_id", "D6"),
            "retained_defect": repair_payload.get("defect_id", "D6"),
            "across_setting_inference": "descriptive_only_single_retained_setting",
        },
        "repair": {
            "selection": repair_payload.get("selection", {}).get("metrics"),
            "feedback_packet": repair_payload.get("feedback_packet"),
            "localizer_rankings": repair_payload.get("localizer_rankings"),
        },
        "final_test": {
            "elapsed_s": final_payload.get("elapsed_s"),
            "arms": table,
        },
    }
