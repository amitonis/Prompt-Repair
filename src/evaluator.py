from __future__ import annotations

import json
from typing import Any

FIELDS = ("event", "date", "venue")
MISSING = object()


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonstandard JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _schema_issues(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["TOP_LEVEL_NOT_OBJECT"]
    issues: list[str] = []
    if set(value) != set(FIELDS):
        issues.append("SCHEMA_KEYS")
    for field in FIELDS:
        if field not in value:
            continue
        item = value[field]
        if item is None and field != "event":
            continue
        if not isinstance(item, str):
            issues.append("WRONG_TYPE")
        elif not item.strip():
            issues.append("EMPTY_STRING")
    return list(dict.fromkeys(issues))


def _matches(actual: Any, expected: str | None) -> bool:
    if actual is MISSING:
        return False
    if expected is None:
        return actual is None
    if not isinstance(actual, str):
        return False
    # Whitespace normalization only: preserve spelling, case, and date wording.
    return " ".join(actual.split()) == " ".join(expected.split())


def score_response(raw: str, gold: dict[str, str | None]) -> dict[str, Any]:
    """Score one completed model response, without cleaning or repairing it.

    Transport failures should be recorded by the runner separately.
    """
    if not isinstance(raw, str):
        raise TypeError("raw must be a string")
    if _schema_issues(gold):
        raise ValueError("Reference answer violates the project schema")

    result: dict[str, Any] = {
        "json_valid": False,
        "schema_valid": False,
        "field_correct": {field: False for field in FIELDS},
        "record_correct": False,
        "issues": [],
        "parsed": None,
    }
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, RecursionError) as exc:
        result["issues"] = ["MALFORMED_JSON"]
        result["parse_error"] = str(exc)
        return result

    result["json_valid"] = True
    result["parsed"] = parsed
    issues = _schema_issues(parsed)
    result["schema_valid"] = not issues

    if isinstance(parsed, dict):
        for field in FIELDS:
            actual = parsed.get(field, MISSING)
            correct = _matches(actual, gold[field])
            result["field_correct"][field] = correct
            if correct or actual is MISSING:
                continue
            if gold[field] is None:
                issues.append("NULL_POLICY_VIOLATION")
            elif actual is None:
                issues.append("OMITTED_SUPPORTED_VALUE")
            else:
                issues.append(f"{field.upper()}_MISMATCH")

    result["record_correct"] = (
        result["schema_valid"] and all(result["field_correct"].values())
    )
    result["issues"] = list(dict.fromkeys(issues))
    return result


def levenshtein_chars(left: str, right: str) -> int:
    """Return character insertion/deletion/substitution distance."""
    if not isinstance(left, str) or not isinstance(right, str):
        raise TypeError("Both arguments must be strings")
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]
