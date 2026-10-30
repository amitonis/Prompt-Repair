"""Structural validation of the 60-notice dataset (analysis_plan.md Section 2).

Run from the repository root: ``python scripts/validate_data.py``.
Exits nonzero if any check fails. Prints category counts per split and the
characters following every gold event string (diagnostic for span choices).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIELDS = ("event", "date", "venue")
SPLITS = ("feedback", "selection", "test")
EXPECTED_SIZES = {"feedback": 10, "selection": 10, "test": 40}

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def load_jsonl(name: str) -> list[dict]:
    path = DATA / name
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def collapse(text: str) -> str:
    return " ".join(text.split())


def main() -> int:
    feedback = load_jsonl("feedback.jsonl")
    selection = load_jsonl("selection.jsonl")
    inputs = load_jsonl("test_inputs.jsonl")
    labels = {r["id"]: r for r in load_jsonl("test_labels.jsonl")}
    manifest = json.loads((DATA / "split_manifest.json").read_text(encoding="utf-8"))

    records: list[dict] = []
    for record in feedback + selection:
        records.append(record)
    for record in inputs:
        label = labels.get(record["id"])
        check(label is not None, f"{record['id']}: missing test label")
        records.append({**record, "gold": (label or {}).get("gold"),
                        "evidence": (label or {}).get("evidence"),
                        "annotation_notes": (label or {}).get("annotation_notes", {})})

    # 1. Counts and unique ids.
    check(len(records) == 60, f"expected 60 records, found {len(records)}")
    ids = [r["id"] for r in records]
    check(len(set(ids)) == len(ids), "duplicate record ids")
    for split, expected in EXPECTED_SIZES.items():
        actual = sum(1 for r in records if r["split"] == split)
        check(actual == expected, f"{split}: {actual} records, expected {expected}")
    check(set(labels) == {r["id"] for r in inputs},
          "test label ids do not match test input ids")
    check(manifest["status"] in {"planned", "review_pending", "frozen"},
          f"unknown manifest status {manifest['status']!r}")

    # 2. Manifest matches data files and planned counts.
    listed = {r["id"]: r for r in manifest["records"]}
    check(set(listed) == set(ids), "manifest record ids differ from data files")
    for record in records:
        entry = listed.get(record["id"])
        if entry is None:
            continue
        for key in ("family_id", "split", "category"):
            check(entry[key] == record[key],
                  f"{record['id']}: manifest {key}={entry[key]!r}, data={record[key]!r}")
    for category, counts in manifest["planned_counts"].items():
        for split in SPLITS:
            actual = sum(1 for r in records
                         if r["category"] == category and r["split"] == split)
            check(actual == counts[split],
                  f"planned {category}/{split}={counts[split]}, found {actual}")

    # 3. Gold schema and types.
    for record in records:
        gold = record.get("gold")
        check(isinstance(gold, dict), f"{record['id']}: gold is not an object")
        if not isinstance(gold, dict):
            continue
        check(set(gold) == set(FIELDS),
              f"{record['id']}: gold keys {sorted(gold)} != {list(FIELDS)}")
        for field in FIELDS:
            value = gold.get(field)
            if value is None:
                check(field != "event", f"{record['id']}: event is null")
            else:
                check(isinstance(value, str), f"{record['id']}.{field}: not a string")
                check(not isinstance(value, str) or value.strip(),
                      f"{record['id']}.{field}: empty string")

    # 4. Evidence equals gold.
    for record in records:
        check(record.get("evidence") == record.get("gold"),
              f"{record['id']}: evidence differs from gold")

    # 5. Non-null gold appears verbatim in the notice.
    for record in records:
        notice = collapse(record["notice"])
        for field in FIELDS:
            value = (record.get("gold") or {}).get(field)
            if value:
                check(collapse(value) in notice,
                      f"{record['id']}.{field}: gold not found verbatim in notice")

    # 6. Every null has an annotation note.
    for record in records:
        notes = record.get("annotation_notes") or {}
        check(set(notes) <= set(FIELDS),
              f"{record['id']}: unexpected annotation note keys {sorted(notes)}")
        for field in FIELDS:
            if (record.get("gold") or {}).get(field) is None:
                check(bool(str(notes.get(field, "")).strip()),
                      f"{record['id']}.{field}: null without annotation note")

    # 7. Families never cross splits.
    family_splits: dict[str, set[str]] = {}
    for record in records:
        family_splits.setdefault(record["family_id"], set()).add(record["split"])
    crossed = sorted(f for f, s in family_splits.items() if len(s) > 1)
    check(not crossed, f"families span splits: {crossed}")

    # 8. No duplicate event names.
    events = Counter(collapse(r["gold"]["event"]) for r in records if r.get("gold"))
    duplicates = sorted(name for name, count in events.items() if count > 1)
    check(not duplicates, f"duplicate event names: {duplicates}")

    # 9. No whitespace anomalies anywhere.
    for record in records:
        texts = [("notice", record["notice"])]
        for field in FIELDS:
            value = (record.get("gold") or {}).get(field)
            if value:
                texts.append((f"gold.{field}", value))
        for name, text in texts:
            problems = []
            if "  " in text:
                problems.append("double space")
            if "\t" in text:
                problems.append("tab")
            if "\n" in text or "\r" in text:
                problems.append("newline")
            if text != text.strip():
                problems.append("leading/trailing space")
            check(not problems,
                  f"{record['id']}.{name}: whitespace anomalies {problems}")

    # Diagnostic output.
    print("Category counts per split")
    for split in SPLITS:
        counts = Counter(r["category"] for r in records if r["split"] == split)
        print(f"  {split:9s} {dict(sorted(counts.items()))}")
    print(f"\nRecords: {len(records)}  Families: {len(family_splits)}  "
          f"Manifest status: {manifest['status']}")
    nulls = Counter()
    for record in records:
        for field in FIELDS:
            if (record.get("gold") or {}).get(field) is None:
                nulls[field] += 1
    print(f"Null gold: {dict(sorted(nulls.items()))}")

    print("\nCharacters following each gold event string")
    for record in records:
        notice = collapse(record["notice"])
        event = collapse(record["gold"]["event"])
        index = notice.find(event)
        tail = notice[index + len(event): index + len(event) + 20]
        print(f"  {record['id']}  {event!r}  -> {tail!r}")

    if failures:
        print(f"\nFAILED: {len(failures)} check(s)")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("\nPASSED: all dataset checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
