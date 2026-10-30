"""Create a reproducible case-level analysis of the main D6 test errors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FINAL_TEST = RUNS / "d6_final_test.json"
RESTRICTED_TEST = RUNS / "restricted_baseline_final_test.json"
RESTRICTED_DEVELOPMENT = RUNS / "restricted_baseline_development.json"
TEST_INPUTS = ROOT / "data" / "test_inputs.jsonl"
TEST_LABELS = ROOT / "data" / "test_labels.jsonl"
OUTPUT = RUNS / "ERROR_ANALYSIS.md"
ARMS = ("rewrite", "oracle", "restricted_baseline")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("|", "\\|").replace("\n", " ")


def build_analysis() -> str:
    final_payload = json.loads(FINAL_TEST.read_text(encoding="utf-8"))
    restricted_payload = json.loads(RESTRICTED_TEST.read_text(encoding="utf-8"))
    development_payload = json.loads(RESTRICTED_DEVELOPMENT.read_text(encoding="utf-8"))
    input_by_id = {row["id"]: row for row in _jsonl(TEST_INPUTS)}
    gold_by_id = {row["id"]: row["gold"] for row in _jsonl(TEST_LABELS)}
    records_by_arm = {
        **{
            name: {row["id"]: row for row in final_payload["results"][name]["records"]}
            for name in ("rewrite", "oracle")
        },
        "restricted_baseline": {
            row["id"]: row for row in restricted_payload["records"]
        },
    }

    category_order = list(dict.fromkeys(row["category"] for row in input_by_id.values()))
    lines = [
        "# D6 final-test error analysis",
        "",
        "Generated from the frozen test labels and saved final-test outputs. "
        "This analysis does not change any labels or strict scores.",
        "",
        "## Accuracy by category",
        "",
        "| Category | Rewrite | Oracle | Restricted baseline |",
        "| --- | ---: | ---: | ---: |",
    ]
    for category in category_order:
        cells = []
        category_ids = [
            notice_id
            for notice_id, row in input_by_id.items()
            if row["category"] == category
        ]
        for arm in ARMS:
            records = records_by_arm[arm]
            correct = sum(bool(records[notice_id]["score"]["record_correct"]) for notice_id in category_ids)
            cells.append(f"{correct}/{len(category_ids)}")
        lines.append(f"| {category} | " + " | ".join(cells) + " |")

    error_ids = [
        notice_id
        for notice_id, record in records_by_arm["oracle"].items()
        if not record["score"]["record_correct"]
    ]
    shared = all(
        {notice_id for notice_id, record in records_by_arm[arm].items()
         if not record["score"]["record_correct"]} == set(error_ids)
        for arm in ARMS
    )
    lines.extend([
        "",
        "All three listed arms scored 35/40 (87.5%) and had valid JSON and schema "
        "on all 40 notices. They scored 26/26 on venue-absent/pending, date-absent, "
        "deadline-distractor, and explicit-correction cases. Their remaining errors "
        "were concentrated in complete event names and unresolved date conflicts.",
        "",
        "## Incorrect cases",
        "",
        "| ID | Category | Field / issue | Expected | Rewrite output | Oracle output | Restricted-baseline output |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for notice_id in error_ids:
        gold = gold_by_id[notice_id]
        oracle = records_by_arm["oracle"][notice_id]
        restricted = records_by_arm["restricted_baseline"][notice_id]
        rewrite = records_by_arm["rewrite"][notice_id]
        wrong_fields = [
            field for field, correct in oracle["score"]["field_correct"].items()
            if not correct
        ]
        issue = ", ".join(oracle["score"]["issues"])
        expected = {field: gold[field] for field in wrong_fields}
        values = [
            notice_id,
            input_by_id[notice_id]["category"],
            f"{', '.join(wrong_fields)} ({issue})",
            _cell(expected),
            _cell(rewrite["score"]["parsed"]),
            _cell(oracle["score"]["parsed"]),
            _cell(restricted["score"]["parsed"]),
        ]
        lines.append("| " + " | ".join(values) + " |")

    development_records = development_payload["records"]
    split_totals: dict[str, list[int]] = {}
    for record in development_records:
        counts = split_totals.setdefault(record["split"], [0, 0])
        counts[0] += bool(record["score"]["record_correct"])
        counts[1] += 1
    development_errors = [
        record for record in development_records
        if not record["score"]["record_correct"]
    ]
    dev_labels = {
        row["id"]: row["gold"]
        for path in (ROOT / "data" / "feedback.jsonl", ROOT / "data" / "selection.jsonl")
        for row in _jsonl(path)
        if "gold" in row
    }
    development_failure_rows = [
        f"- `{record['id']}` ({record['split']}): expected event "
        f"{_cell(dev_labels[record['id']]['event'])}; observed "
        f"{_cell(record['score']['parsed'].get('event'))}."
        for record in development_errors
    ]

    lines.extend([
        "",
        "## Development-set signal and protocol boundary",
        "",
        f"The restricted-baseline development result was "
        f"{split_totals['feedback'][0]}/{split_totals['feedback'][1]} on feedback "
        f"and {split_totals['selection'][0]}/{split_totals['selection'][1]} on selection. "
        "Both development errors were selection cases:",
        *development_failure_rows,
        "The feedback subset had no errors to provide as editor examples. Under the "
        "frozen procedure, the selection subset evaluates candidate acceptance; it "
        "must not be reused to invent or tune the primary prompt. A follow-up that "
        "targets event-name boundaries should use a separately collected and reviewed "
        "exploratory set, with results kept distinct from the preregistered study.",
    ])

    lines.extend([
        "",
        "The rewrite, oracle, and restricted-baseline arms failed the same five IDs: "
        + ", ".join(error_ids) + ("." if shared else "; their error sets differ."),
        "Three complete-event cases (TEST-001, TEST-003, TEST-004) are strict event-name "
        "mismatches involving terminal punctuation or, for the restricted baseline on "
        "TEST-001, an added organizer phrase. The notice text places the punctuation "
        "immediately after the event title, while the frozen gold labels omit it. This "
        "is a possible annotation-boundary ambiguity under the verbatim-copy rule; the "
        "frozen labels and scores are retained as preregistered.",
        "",
        "TEST-037 and TEST-038 contain two conflicting dates without a stated priority. "
        "The gold label requires `null`; the models select one listed date instead. "
        "These are substantive null-policy failures rather than formatting failures.",
        "",
        "## Interpretation",
        "",
        "For these cases, adding more general instructions did not eliminate the "
        "observed punctuation and conflict-resolution errors: the rewrite and oracle "
        "have the same five strict failures, and the restricted baseline has the same "
        "error IDs. The case patterns motivate a separate error-focused extension "
        "and a human audit of the punctuation convention. Do not revise the primary "
        "prompt or labels using these observed selection/test outcomes.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    OUTPUT.write_text(build_analysis(), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
