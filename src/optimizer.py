"""Optimization branches and the frozen acceptance rule (analysis plan, Section 5)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .patcher import (
    ALL_COMPONENTS,
    RESTRICTED_COMPONENTS,
    apply_patch,
    apply_rewrite,
    apply_section_rewrite,
)

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
EDITOR_TEMPLATES = {
    "rewrite": "rewrite_editor.txt",
    "span": "targeted_editor.txt",
    "section": "section_editor.txt",
}
PLACEHOLDERS = ("{{TASK_CONTRACT}}", "{{PROMPT_SECTIONS_JSON}}", "{{FEEDBACK_JSON}}")
SECTION_PLACEHOLDER = "{{SECTION}}"

Ask = Callable[[str], str]
Score = Callable[[dict[str, str]], dict[str, Any]]


@dataclass
class BranchResult:
    sections: dict[str, str]
    accepted: bool
    metrics: dict[str, Any]
    attempts: list[dict[str, Any]]
    stopped_reason: str


@dataclass
class SectionPatchResult:
    """One patch per targeted section, for the patch-all-sections lookups."""

    sections: dict[str, str]
    accepted: bool
    accepted_section: str | None
    metrics: dict[str, Any]
    patches: dict[str, dict[str, Any]]
    attempts: list[dict[str, Any]]
    stopped_reason: str


def should_accept(current: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Frozen acceptance rule (analysis plan, Section 5).

    Accept a candidate only if its selection-set strict accuracy is strictly
    higher than the current prompt's and it introduces no new JSON/schema
    invalidity (invalid-output count must not increase).
    """
    return (
        candidate["selection_records_correct"] > current["selection_records_correct"]
        and candidate["selection_invalid"] <= current["selection_invalid"]
    )


def choose_candidate(current: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the accepted candidate among attempts (earlier attempt wins ties).

    Each metrics dict needs: selection_records_correct, selection_invalid.
    Returns the accepted candidate's metrics, or None if the prompt is
    unchanged (counts as a failed repair, not an exclusion).
    """
    best: dict[str, Any] | None = None
    for candidate in candidates:
        if not should_accept(current, candidate):
            continue
        if best is None or candidate["selection_records_correct"] > best["selection_records_correct"]:
            best = candidate
    return best


def parse_editor_json(raw: str) -> dict[str, Any]:
    """Extract the editor's JSON object, tolerating fences and surrounding text."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("empty editor response")
    text = raw.strip()
    fenced = re.match(r"^```[A-Za-z0-9_-]*\s*(.*?)\s*```$", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    try:
        proposal, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(proposal, dict):
        raise ValueError("editor response is not a JSON object")
    return proposal


def render_editor_prompt(
    kind: str,
    *,
    task_contract: str,
    sections: dict[str, str],
    feedback_packet: dict[str, Any],
    section: str | None = None,
) -> str:
    """Fill one frozen editor template with the contract, prompt and feedback."""
    if kind not in EDITOR_TEMPLATES:
        raise ValueError(f"unknown editor template: {kind}")
    template = (PROMPTS_DIR / EDITOR_TEMPLATES[kind]).read_text(encoding="utf-8")
    required = list(PLACEHOLDERS)
    if SECTION_PLACEHOLDER in template:
        if section not in ALL_COMPONENTS:
            raise ValueError(f"template {kind} needs a valid target section")
        required.append(SECTION_PLACEHOLDER)
    missing = [placeholder for placeholder in required if placeholder not in template]
    if missing:
        raise ValueError(f"template {kind} is missing placeholders: {missing}")
    values = {
        "{{TASK_CONTRACT}}": task_contract.strip(),
        "{{PROMPT_SECTIONS_JSON}}": json.dumps(sections, indent=2, sort_keys=True),
        "{{FEEDBACK_JSON}}": json.dumps(feedback_packet, indent=2, sort_keys=True),
    }
    if SECTION_PLACEHOLDER in template:
        values[SECTION_PLACEHOLDER] = section
    for placeholder, value in values.items():
        template = template.replace(placeholder, value)
    if "{{" in template or "}}" in template:
        raise ValueError("template contains an unfilled placeholder")
    return template


def run_rewrite_branch(
    current_sections: dict[str, str],
    current_metrics: dict[str, Any],
    *,
    ask: Ask,
    score: Score,
    task_contract: str,
    feedback_packet: dict[str, Any],
    max_attempts: int = 2,
    max_prompt_words: int = 240,
) -> BranchResult:
    """Rewrite arm: one complete rewrite per attempt, at most two attempts."""
    return _run_proposal_branch(
        current_sections,
        current_metrics,
        ask=ask,
        score=score,
        task_contract=task_contract,
        feedback_packet=feedback_packet,
        template="rewrite",
        max_attempts=max_attempts,
        apply=lambda sections, proposal: apply_rewrite(
            sections, proposal, max_prompt_words=max_prompt_words
        ),
    )


def run_span_edit_branch(
    current_sections: dict[str, str],
    current_metrics: dict[str, Any],
    *,
    ask: Ask,
    score: Score,
    task_contract: str,
    feedback_packet: dict[str, Any],
    max_attempts: int = 2,
    max_new_text_words: int = 60,
    max_prompt_words: int = 240,
) -> BranchResult:
    """Span edit arm: one constrained span replacement per attempt."""
    return _run_proposal_branch(
        current_sections,
        current_metrics,
        ask=ask,
        score=score,
        task_contract=task_contract,
        feedback_packet=feedback_packet,
        template="span",
        max_attempts=max_attempts,
        apply=lambda sections, proposal: apply_patch(
            sections,
            proposal,
            allowed_components=RESTRICTED_COMPONENTS,
            max_new_text_words=max_new_text_words,
            max_prompt_words=max_prompt_words,
        ),
    )


def _run_proposal_branch(
    current_sections: dict[str, str],
    current_metrics: dict[str, Any],
    *,
    ask: Ask,
    score: Score,
    task_contract: str,
    feedback_packet: dict[str, Any],
    template: str,
    max_attempts: int,
    apply: Callable[[dict[str, str], dict[str, Any]], dict[str, Any]],
) -> BranchResult:
    attempts: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    stopped = "attempts_exhausted"

    for attempt in range(1, max_attempts + 1):
        prompt = render_editor_prompt(
            template,
            task_contract=task_contract,
            sections=current_sections,
            feedback_packet=feedback_packet,
        )
        raw = ask(prompt)
        entry: dict[str, Any] = {"attempt": attempt, "raw": raw}
        try:
            proposal = parse_editor_json(raw)
        except ValueError as exc:
            entry.update({"valid": False, "reason": f"PARSE_ERROR: {exc}"})
            attempts.append(entry)
            continue

        entry["proposal"] = proposal
        result = apply(current_sections, proposal)
        if result.get("no_change"):
            entry.update({"valid": False, "reason": "NO_CHANGE"})
            attempts.append(entry)
            stopped = "no_change"
            break
        if not result["ok"]:
            entry.update({"valid": False, "reason": result["reason"]})
            attempts.append(entry)
            continue
        entry.update({"valid": True, "sections": result["sections"]})
        attempts.append(entry)
        candidates.append(entry)

    best: dict[str, Any] | None = None
    for candidate in candidates:
        metrics = score(candidate["sections"])
        candidate["metrics"] = metrics
        candidate["accepted"] = should_accept(current_metrics, metrics)
        if candidate["accepted"] and (
            best is None
            or metrics["selection_records_correct"] > best["metrics"]["selection_records_correct"]
        ):
            best = candidate

    if best is None:
        if candidates and stopped != "no_change":
            stopped = "no_candidate_accepted"
        return BranchResult(dict(current_sections), False, dict(current_metrics), attempts, stopped)
    return BranchResult(dict(best["sections"]), True, best["metrics"], attempts, "accepted")


def run_targeted_branch(
    current_sections: dict[str, str],
    current_metrics: dict[str, Any],
    *,
    ask: Ask,
    score: Score,
    task_contract: str,
    feedback_packet: dict[str, Any],
    target_sections: tuple[str, ...] = ALL_COMPONENTS,
    max_attempts: int = 2,
    max_new_text_words: int = 60,
    max_prompt_words: int = 240,
) -> SectionPatchResult:
    """Patch-all-sections: one full-section rewrite per targeted section.

    All patches in an attempt share the packet and limits. Attempt 2 runs only
    when no patch was accepted in attempt 1; a section-level ``no_change``
    yields no candidate for that section, and all four declining stops the
    branch.
    """
    if not target_sections:
        raise ValueError("target_sections must not be empty")

    attempts: list[dict[str, Any]] = []
    last_valid: dict[str, dict[str, Any]] = {}
    last_entries: dict[str, dict[str, Any]] = {}
    best_by_section: dict[str, dict[str, Any]] = {}
    retry_notes: dict[str, str] = {}
    stopped = "attempts_exhausted"
    accepted_any = False

    for attempt in range(1, max_attempts + 1):
        log: dict[str, Any] = {"attempt": attempt, "patches": {}}
        candidates: list[tuple[str, dict[str, Any]]] = []
        for name in target_sections:
            prompt = render_editor_prompt(
                "section",
                task_contract=task_contract,
                sections=current_sections,
                feedback_packet=feedback_packet,
                section=name,
            )
            if name in retry_notes:
                prompt += (
                    "\n\nRETRY FEEDBACK FROM THE VALIDATOR:\n"
                    f"Your previous proposal for {name!r} was rejected: "
                    f"{retry_notes[name]} Make a materially corrected proposal. "
                    "For NEW_TEXT_TOO_LONG, keep new_text at or below 60 words."
                )
            raw = ask(prompt)
            entry: dict[str, Any] = {"raw": raw, "valid": False}
            try:
                proposal = parse_editor_json(raw)
            except ValueError as exc:
                entry["reason"] = f"PARSE_ERROR: {exc}"
            else:
                entry["proposal"] = proposal
                result = apply_section_rewrite(
                    current_sections,
                    proposal,
                    name,
                    max_new_text_words=max_new_text_words,
                    max_prompt_words=max_prompt_words,
                )
                if result.get("no_change"):
                    entry["reason"] = "NO_CHANGE"
                elif not result["ok"]:
                    entry["reason"] = result["reason"]
                    retry_notes[name] = result["reason"]
                else:
                    entry.update({"valid": True, "sections": result["sections"]})
                    last_valid[name] = entry
                    candidates.append((name, entry))
            log["patches"][name] = entry
            last_entries[name] = entry

        for name, entry in candidates:
            metrics = score(entry["sections"])
            entry["metrics"] = metrics
            entry["accepted"] = should_accept(current_metrics, metrics)
            if not entry["accepted"]:
                retry_notes[name] = (
                    "the candidate did not improve selection exact-record accuracy "
                    "without increasing invalid outputs"
                )
            if entry["accepted"]:
                accepted_any = True
                incumbent = best_by_section.get(name)
                if (
                    incumbent is None
                    or metrics["selection_records_correct"]
                    > incumbent["metrics"]["selection_records_correct"]
                ):
                    best_by_section[name] = {**entry, "attempt": attempt}
        attempts.append(log)

        if accepted_any:
            stopped = "accepted"
            break
        if all(log["patches"][name].get("reason") == "NO_CHANGE" for name in target_sections):
            stopped = "no_change"
            break

    patches: dict[str, dict[str, Any]] = {}
    for name in target_sections:
        if name in best_by_section:
            patches[name] = best_by_section[name]
        elif name in last_entries:
            patches[name] = {**last_entries[name], "accepted": False}
        else:
            patches[name] = {"accepted": False, "reason": "NO_CANDIDATE"}

    if not best_by_section:
        if stopped != "no_change":
            stopped = "no_candidate_accepted" if last_valid else "attempts_exhausted"
        return SectionPatchResult(
            dict(current_sections), False, None, dict(current_metrics), patches, attempts, stopped
        )

    best_name, best = max(
        best_by_section.items(),
        key=lambda item: item[1]["metrics"]["selection_records_correct"],
    )
    return SectionPatchResult(
        dict(best["sections"]), True, best_name, best["metrics"], patches, attempts, "accepted"
    )
