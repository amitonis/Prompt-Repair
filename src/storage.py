"""Dataset loading, split enforcement, and run-record persistence (Section 13.4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SUBSETS = ("feedback", "selection", "test")
INPUT_FIELDS = ("id", "family_id", "split", "category", "notice")


def load_jsonl(filename: str) -> list[dict[str, Any]]:
    records = []
    with (DATA_DIR / filename).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_subset(split: str, with_labels: bool = True) -> list[dict[str, Any]]:
    """Load one subset. Test labels are kept out of the extraction-request path."""
    if split not in SUBSETS:
        raise ValueError(f"Unknown split: {split}")
    if split == "feedback":
        return load_jsonl("feedback.jsonl")
    if split == "selection":
        return load_jsonl("selection.jsonl")
    inputs = load_jsonl("test_inputs.jsonl")
    if not with_labels:
        return inputs
    labels = {r["id"]: r for r in load_jsonl("test_labels.jsonl")}
    for record in inputs:
        if record["id"] not in labels:
            raise ValueError(f"Missing label for test case {record['id']}")
        record["gold"] = labels[record["id"]]["gold"]
    return inputs


def check_family_isolation(records: list[dict[str, Any]]) -> list[str]:
    """Every family_id must live in exactly one split (Section 6.3)."""
    family_splits: dict[str, set[str]] = {}
    for record in records:
        family_splits.setdefault(record["family_id"], set()).add(record["split"])
    return sorted(f for f, splits in family_splits.items() if len(splits) > 1)


def save_run_artifacts(run_id: str, artifacts: dict[str, Any]) -> None:
    """Persist runs/RUN_ID/ records; mirror to Drive or GitHub after each batch."""
    raise NotImplementedError("Activated with the first real sweep (Day 8)")
