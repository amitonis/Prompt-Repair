"""Dataset loading, split enforcement, and run-record persistence (Section 13.4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
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
    """Persist a run's JSON-compatible artifacts under ``runs/<run_id>``.

    Mapping values are written as JSON. Lists of mappings whose filename ends
    in ``.jsonl`` are written one record per line, which keeps generation and
    score logs append-friendly and easy to resume. This function deliberately
    does not perform cloud synchronization; the caller should sync the run
    directory after each major stage.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a nonempty string")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single safe directory name")
    if not isinstance(artifacts, dict):
        raise TypeError("artifacts must be a dict")

    destination = RUNS_DIR / run_id
    destination.mkdir(parents=True, exist_ok=True)
    for filename, value in artifacts.items():
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("artifact filenames must be nonempty strings")
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe artifact path: {filename}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".jsonl":
            if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
                raise ValueError(f"JSONL artifact {filename} must be a list of objects")
            payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in value)
        else:
            payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
        target.write_text(payload, encoding="utf-8")
