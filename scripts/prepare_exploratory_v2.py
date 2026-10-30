"""Derive an exploratory v2 from the reviewed draft without touching draft 01."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "exploratory_draft_01"
DESTINATION = ROOT / "data" / "exploratory_draft_02"
EXPECTED_HASHES = {
    "inputs.jsonl": "3e45ffca92c9506c26091c5008e29e6f2133aec2454f84c8da1b188ebc5f1445",
    "labels_draft.jsonl": "ec5fa4ddce50bf10c0d0c2b60536bda1ed1b04ebceee506f4ff43d0be2b51e77",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(content, encoding="utf-8")


def main() -> int:
    if DESTINATION.exists():
        raise FileExistsError(
            f"Refusing to overwrite reviewed or partially reviewed v2 at {DESTINATION}."
        )
    for name, expected in EXPECTED_HASHES.items():
        actual = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"Refusing to derive v2: {name} hash mismatch ({actual})."
            )

    inputs = _load_jsonl(SOURCE / "inputs.jsonl")
    labels = _load_jsonl(SOURCE / "labels_draft.jsonl")
    for row in inputs:
        row["split"] = "exploratory_v2"
    input_by_id = {row["id"]: row for row in inputs}
    label_by_id = {row["id"]: row for row in labels}

    input_by_id["EXP-EB-004"]["notice"] = (
        'The event is named "Ink and Iris Workshop". The Creative Arts Circle '
        "will host it on 15 March 2027 in North Gallery Studio."
    )
    input_by_id["EXP-UC-003"]["notice"] = (
        "The Film Poster Workshop is listed in Studio A on the department "
        "webpage and in Studio C on the club flyer. Neither source says the "
        "venue changed. Date: 24 March 2027."
    )

    eb004 = label_by_id["EXP-EB-004"]
    eb004["gold"]["event"] = "Ink and Iris Workshop"
    eb004["gold"]["venue"] = "North Gallery Studio"
    eb004["evidence"]["event"] = "Ink and Iris Workshop"
    eb004["evidence"]["venue"] = "North Gallery Studio"
    eb004["annotation_notes"]["event"] = (
        "Gold updated with the renamed event in exploratory v2. Quotation "
        "marks are treated as delimiters; this punctuation convention is not "
        "specified by the contract and must be reported separately."
    )

    DESTINATION.mkdir(parents=True, exist_ok=True)
    _write_jsonl(DESTINATION / "inputs.jsonl", inputs)
    _write_jsonl(DESTINATION / "labels_draft.jsonl", labels)

    (DESTINATION / "REVIEW_GUIDE.md").write_text(
        """# Exploratory draft 02 — human review required

Status: **DRAFT — DO NOT RUN UNTIL HUMAN SIGN-OFF IS COMPLETED.**

This v2 derives from draft 01. Draft 01 is preserved unchanged. The proposed
labels are review-only and are separate from the frozen 60-case dataset and
primary study results.

## Changes from draft 01

- EXP-EB-004 now uses a distinct event, organizer, and venue to avoid reusing
  the frozen SEL-006 entities. Its event and venue gold/evidence were updated
  to match the revised notice.
- EXP-UC-003 no longer calls one source "current"; its gold label remains
  `venue: null` because neither listed source has priority.
- All other notices and gold labels carry over unchanged.

## Review decisions to confirm

1. Review all 12 labels, including the revised event and venue in EXP-EB-004.
2. Confirm whether quotation marks in EXP-EB-004 delimit the title. Contract
   text does not define quotation-mark handling. Record this case separately.
3. Confirm the terminal-exclamation convention in EXP-EB-005. The proposed
   gold excludes `!` to match the A1 convention, which is not explicit in the
   contract. Report EB-005 separately from the other event-boundary cases.
4. Check that the new EB-004 entities do not collide with any frozen event
   name, and that all evidence matches the revised notice.
5. Confirm every unresolved conflict and its null label.

## Exploratory reporting limits

There are 12 cases. Report counts only, mark all results **post hoc and
exploratory**, and keep them separate from the primary results. Five of six
conflict cases concern dates and one concerns venue. Every conflict notice
explicitly says neither source has priority, so these examples are easier than
real-world conflicts. Do not claim general performance from this small set.

The human reviewer must complete `human_review.json` and sign before any model
run. The advisory Claude pass is recorded under draft 01; it is not a human
signature.
""",
        encoding="utf-8",
    )
    (DESTINATION / "CHANGELOG.md").write_text(
        """# Exploratory v2 change log

- Source: `data/exploratory_draft_01/`.
- Source input SHA-256: `3e45ffca92c9506c26091c5008e29e6f2133aec2454f84c8da1b188ebc5f1445`.
- Source label-draft SHA-256: `ec5fa4ddce50bf10c0d0c2b60536bda1ed1b04ebceee506f4ff43d0be2b51e77`.
- EXP-EB-004 notice entities changed to avoid overlap with frozen SEL-006;
  corresponding event and venue gold/evidence updated as authorized in chat.
- EXP-UC-003 wording changed by removing "current"; labels unchanged.
- No primary data, prompts, labels, or scores were modified.
- Status: draft; human label review and sign-off are still required before any run.
""",
        encoding="utf-8",
    )
    human_review = {
        "dataset": "exploratory_draft_02",
        "status": "review_pending",
        "human_reviewer": "TO_FILL",
        "sign_off_date": "TO_FILL",
        "human_decisions": {row["id"]: "TO_FILL" for row in labels},
        "run_authorized": False,
    }
    (DESTINATION / "human_review.json").write_text(
        json.dumps(human_review, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared review-pending v2 at {DESTINATION}")
    for name in ("inputs.jsonl", "labels_draft.jsonl"):
        digest = hashlib.sha256((DESTINATION / name).read_bytes()).hexdigest()
        print(f"{name} SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
