"""Fault localizers (analysis plan, Sections 4 and 6).

Frozen set: L-LLM, L-Rule, L-Behav. Each localizer outputs a ranked list of
the four prompt sections. Tie-break order is fixed: output_format,
constraints, instruction, context. Localizers use feedback-set data only
(or no data, for L-Rule); selection and test data are never used.

L-Probe is exploratory (analysis_plan.md Section 6): it scores the four
already-generated section patches on the feedback set and is reported outside
the frozen set used for P1.
"""

from __future__ import annotations

from typing import Any

from .defects import SECTIONS, load_prompt

TIE_BREAK_ORDER = ("output_format", "constraints", "instruction", "context")

ERROR_CODE_TO_SECTION = {
    "MALFORMED_JSON": "output_format",
    "SCHEMA_KEYS": "output_format",
    "WRONG_TYPE": "output_format",
    "EMPTY_STRING": "constraints",
    "EVENT_MISMATCH": "constraints",
    "DATE_MISMATCH": "constraints",
    "VENUE_MISMATCH": "constraints",
    "NULL_POLICY_VIOLATION": "constraints",
    "OMITTED_SUPPORTED_VALUE": "constraints",
}


def rule_localizer(feedback_packet: dict[str, Any]) -> list[str]:
    """L-Rule: map observed error codes to sections, weighted by error counts."""
    weights = {section: 0 for section in TIE_BREAK_ORDER}
    for code, count in feedback_packet.get("error_counts", {}).items():
        section = ERROR_CODE_TO_SECTION.get(code)
        if section is not None:
            weights[section] += count
    return sorted(TIE_BREAK_ORDER, key=lambda s: (-weights[s], TIE_BREAK_ORDER.index(s)))


def behav_neutralize(sections: dict[str, str], section: str) -> dict[str, str]:
    """L-Behav neutralization: replace one section with the matching section
    of the frozen naturalistic original prompt (`prompts/original.json`)."""
    if section not in SECTIONS:
        raise ValueError(f"Unknown section: {section}")
    neutralized = dict(sections)
    neutralized[section] = load_prompt("original")[section]
    return neutralized


def behav_localizer_rank(output_changes: dict[str, list[bool]]) -> list[str]:
    """L-Behav ranking: sections by the share of failing feedback notices
    whose raw output changes when that section is neutralized. Ties use the
    fixed tie-break order. `output_changes[section][i]` is True if the raw
    output of failing notice i changed under the neutralized prompt."""
    def changed_fraction(section: str) -> float:
        changes = output_changes.get(section, [])
        return sum(changes) / len(changes) if changes else 0.0

    return sorted(
        TIE_BREAK_ORDER,
        key=lambda s: (-changed_fraction(s), TIE_BREAK_ORDER.index(s)),
    )


def llm_localizer(feedback_packet: dict[str, Any], prompt_sections: dict[str, str]) -> list[str]:
    """L-LLM: the editor model names the section(s) most likely at fault."""
    raise NotImplementedError("Requires one editor LLM call (Day 7)")


def grad_localizer(*args: Any, **kwargs: Any) -> list[str]:
    """L-Grad: gradient-based token importance per section (Phase 2; not in
    the frozen localizer set)."""
    raise NotImplementedError("Phase 2 extension")
