"""Summarize frozen primary test outputs by field and error type (no inference)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FIELDS = ("event", "date", "venue")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mismatch_type(field: str, expected: Any, actual: Any, parsed: Any) -> str:
    if parsed is None:
        return "unparseable_response"
    if actual is None:
        return "null_or_missing_value"
    if isinstance(expected, str) and isinstance(actual, str):
        if expected in actual:
            return "expected_text_with_extra_content"
        if actual in expected:
            return "truncated_or_partial_text"
    if expected is None and actual is not None:
        return "value_returned_where_null_required"
    if expected is not None and actual is None:
        return "null_returned_where_value_required"
    return f"other_{field}_mismatch"


def summarize_arm(name: str, records: list[dict[str, Any]], gold: dict[str, dict[str, Any]]) -> dict[str, Any]:
    issues: Counter[str] = Counter()
    mismatch_types: dict[str, Counter[str]] = {field: Counter() for field in FIELDS}
    field_correct = Counter()
    field_correct_schema_valid = Counter()
    by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "attempted": 0, "json_valid": 0, "schema_valid": 0}
    )
    incorrect: list[dict[str, Any]] = []
    for row in records:
        score = row["score"]
        case_id = row["id"]
        expected = gold[case_id]["gold"]
        parsed = score.get("parsed")
        actual = parsed if isinstance(parsed, dict) else {}
        issues.update(score.get("issues", []))
        category = row.get("category", "unknown")
        bucket = by_category[category]
        bucket["attempted"] += 1
        bucket["correct"] += int(bool(score.get("record_correct")))
        bucket["json_valid"] += int(bool(score.get("json_valid")))
        bucket["schema_valid"] += int(bool(score.get("schema_valid")))
        wrong_fields = []
        for field in FIELDS:
            correct = bool(score.get("field_correct", {}).get(field, False))
            field_correct[field] += int(correct)
            if score.get("schema_valid"):
                field_correct_schema_valid[field] += int(correct)
            if not correct:
                mismatch_types[field][mismatch_type(field, expected.get(field), actual.get(field), parsed)] += 1
                wrong_fields.append(field)
        if not score.get("record_correct"):
            incorrect.append({
                "id": case_id,
                "family_id": row.get("family_id"),
                "category": category,
                "issues": score.get("issues", []),
                "wrong_fields": wrong_fields,
                "expected": expected,
                "observed": parsed,
            })

    schema_valid_n = sum(bool(row["score"].get("schema_valid")) for row in records)
    return {
        "attempted": len(records),
        "record_correct": sum(bool(row["score"].get("record_correct")) for row in records),
        "json_valid": sum(bool(row["score"].get("json_valid")) for row in records),
        "schema_valid": schema_valid_n,
        "field_correct_strict": {field: field_correct[field] for field in FIELDS},
        "field_correct_among_schema_valid": {field: field_correct_schema_valid[field] for field in FIELDS},
        "issue_counts": dict(issues),
        "field_mismatch_types": {field: dict(counts) for field, counts in mismatch_types.items()},
        "by_category": dict(by_category),
        "incorrect_cases": incorrect,
    }


def main() -> int:
    labels = {row["id"]: row for row in read_jsonl(ROOT / "data" / "test_labels.jsonl")}
    d6 = json.loads((RUNS / "d6_final_test.json").read_text(encoding="utf-8"))
    original = json.loads((RUNS / "original_final_test.json").read_text(encoding="utf-8"))
    restricted = json.loads((RUNS / "restricted_baseline_final_test.json").read_text(encoding="utf-8"))

    all_arms: dict[str, list[dict[str, Any]]] = {
        name: payload["records"] for name, payload in d6["results"].items()
    }
    all_arms["original"] = original["records"]
    all_arms["restricted_baseline"] = restricted["records"]
    summaries = {name: summarize_arm(name, rows, labels) for name, rows in all_arms.items()}

    shared_arms = ("rewrite", "oracle", "restricted_baseline")
    shared_failures = set(x["id"] for x in summaries[shared_arms[0]]["incorrect_cases"])
    for arm in shared_arms[1:]:
        shared_failures &= {x["id"] for x in summaries[arm]["incorrect_cases"]}

    payload = {
        "scope": "offline_descriptive_reanalysis_of_frozen_primary_test_outputs",
        "warning": "This analysis does not change frozen labels, prompts, or primary scores and is not a new evaluation.",
        "notices": 40,
        "families": len({row["family_id"] for row in read_jsonl(ROOT / "data" / "test_inputs.jsonl")}),
        "field_metric_note": (
            "Field exactness follows the existing scorer and is counted independently by field; a correct field may occur in a record with another wrong field or an invalid schema. "
            "JSON validity, schema validity, and exact-record correctness are separate metrics. Conditional field counts are also shown among schema-valid records."
        ),
        "shared_error_ids_rewrite_oracle_restricted": sorted(shared_failures),
        "arms": summaries,
    }
    (RUNS / "field_diagnostics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Frozen primary-test field diagnostics",
        "",
        "Offline descriptive reanalysis of saved primary outputs. No labels, prompts, or primary scores were changed.",
        "",
        "Exact field matches follow the existing evaluator and are independent of the other fields and schema validity. JSON validity, schema validity, exact-record correctness, and field exactness are reported separately. The conditional count uses schema-valid records only.",
        "",
        "| Arm | Record correct | JSON valid | Schema valid | Event exact | Date exact | Venue exact |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in summaries.items():
        n = result["attempted"]
        f = result["field_correct_strict"]
        lines.append(
            f"| {name} | {result['record_correct']}/{n} | {result['json_valid']}/{n} | "
            f"{result['schema_valid']}/{n} | {f['event']}/{n} | {f['date']}/{n} | {f['venue']}/{n} |"
        )
    lines += [
        "",
        f"Shared incorrect IDs for rewrite, oracle, and restricted baseline: {', '.join(sorted(shared_failures)) or 'none'}.",
        "",
        "## Error groups",
        "",
    ]
    for name, result in summaries.items():
        if not result["incorrect_cases"]:
            continue
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| ID | Category | Wrong fields | Issue codes | Expected | Observed |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for case in result["incorrect_cases"]:
            expected = json.dumps(case["expected"], ensure_ascii=False)
            observed = json.dumps(case["observed"], ensure_ascii=False)
            lines.append(
                f"| {case['id']} | {case['category']} | {', '.join(case['wrong_fields'])} | "
                f"{', '.join(case['issues'])} | `{expected}` | `{observed}` |"
            )
        lines.append("")
    lines += [
        "## Reading the field counts",
        "",
        "Field totals are diagnostic views of the same scorer, not additional independent outcomes. A record can have multiple correct fields and still fail exact-record accuracy or schema validation. The primary study remains scored exactly as preregistered.",
        "",
    ]
    (RUNS / "FIELD_DIAGNOSTICS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {RUNS / 'field_diagnostics.json'}")
    print(f"Wrote {RUNS / 'FIELD_DIAGNOSTICS.md'}")
    print(f"Shared error IDs: {', '.join(sorted(shared_failures))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
