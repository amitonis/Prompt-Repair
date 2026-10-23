"""Patch guard: validate and apply one constrained targeted edit (Section 8.6)."""

from __future__ import annotations

import re
from typing import Any

from .prompt_renderer import prompt_word_count

RESTRICTED_COMPONENTS = ("constraints", "output_format")
ALL_COMPONENTS = ("instruction", "context", "constraints", "output_format")
ALLOWED_FIELDS = ("event", "date", "venue")
REQUIRED_PROPOSAL_KEYS = ("action", "component", "old_text", "new_text", "error_codes", "explanation")
FIELD_LIKE = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"')


def _reason(code: str, detail: str = "") -> dict[str, Any]:
    return {"ok": False, "reason": code, "detail": detail, "sections": None}


def _mentions_new_field(new_text: str) -> str | None:
    for token in FIELD_LIKE.findall(new_text):
        if token not in ALLOWED_FIELDS and token.lower() not in ("json", "null"):
            return token
    return None


def apply_patch(
    sections: dict[str, str],
    proposal: dict[str, Any],
    allowed_components: tuple[str, ...] = RESTRICTED_COMPONENTS,
    max_new_text_words: int = 60,
    max_prompt_words: int = 240,
) -> dict[str, Any]:
    """Validate one targeted-edit proposal and apply it to a copy of the prompt.

    Returns {"ok": bool, "reason": str | None, "sections": dict | None}.
    A rejected patch leaves the current prompt unchanged.
    """
    if not isinstance(proposal, dict):
        return _reason("PROPOSAL_NOT_OBJECT")
    missing = [k for k in REQUIRED_PROPOSAL_KEYS if k not in proposal]
    if missing:
        return _reason("MISSING_PROPOSAL_KEYS", ", ".join(missing))
    if proposal.get("action") not in ("replace", "no_change"):
        return _reason("UNKNOWN_ACTION", str(proposal.get("action")))
    if proposal["action"] == "no_change":
        return {"ok": True, "reason": None, "sections": dict(sections), "no_change": True}

    component = proposal["component"]
    if component not in allowed_components:
        return _reason("COMPONENT_NOT_ALLOWED", str(component))
    old_text = proposal["old_text"]
    new_text = proposal["new_text"]
    if not isinstance(old_text, str) or not old_text.strip():
        return _reason("OLD_TEXT_EMPTY")
    if not isinstance(new_text, str) or not new_text.strip():
        return _reason("NEW_TEXT_EMPTY")
    if old_text == new_text:
        return _reason("NO_OP_EDIT")
    occurrences = sections[component].count(old_text)
    if occurrences == 0:
        return _reason("OLD_TEXT_ABSENT")
    if occurrences > 1:
        return _reason("OLD_TEXT_AMBIGUOUS", f"occurs {occurrences} times")
    if len(new_text.split()) > max_new_text_words:
        return _reason("NEW_TEXT_TOO_LONG", f"{len(new_text.split())} words")
    new_field = _mentions_new_field(new_text)
    if new_field:
        return _reason("NEW_FIELD_INTRODUCED", new_field)

    patched_section = sections[component].replace(old_text, new_text, 1)
    patched = dict(sections)
    patched[component] = patched_section
    if prompt_word_count(patched) > max_prompt_words:
        return _reason("PROMPT_TOO_LONG", f"{prompt_word_count(patched)} words")
    return {"ok": True, "reason": None, "sections": patched, "no_change": False}
