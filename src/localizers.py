"""Fault localizers (Section 9A.5).

Each localizer outputs a ranked list of the four prompt sections for a given
prompt and feedback packet. Tie-break order is fixed in advance:
output_format, constraints, instruction, context.
"""

from __future__ import annotations

from typing import Any

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


def llm_localizer(feedback_packet: dict[str, Any], prompt_sections: dict[str, str]) -> list[str]:
    """L-LLM: the editor model names the section(s) most likely at fault."""
    raise NotImplementedError("Requires one editor LLM call (Day 7)")


def ablate_localizer(*args: Any, **kwargs: Any) -> list[str]:
    """L-Ablate: rank sections by accuracy drop when each is removed."""
    raise NotImplementedError("Requires four feedback evaluations (Phase 2)")


def probe_localizer(*args: Any, **kwargs: Any) -> list[str]:
    """L-Probe: request one patch per section, rank by feedback-subset score."""
    raise NotImplementedError("Requires four editor calls and four evaluations (Day 7)")


def grad_localizer(*args: Any, **kwargs: Any) -> list[str]:
    """L-Grad: gradient-based token importance per section (Phase 2)."""
    raise NotImplementedError("Requires a white-box Transformers model (Phase 2)")
