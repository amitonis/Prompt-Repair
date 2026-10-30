"""Patch guard: validate and apply one constrained targeted edit (Section 8.6)."""

from __future__ import annotations

import re
from typing import Any

from .prompt_renderer import prompt_word_count

RESTRICTED_COMPONENTS = ("constraints", "output_format")
ALL_COMPONENTS = ("instruction", "context", "constraints", "output_format")
ALLOWED_FIELDS = ("event", "date", "venue")
REQUIRED_PROPOSAL_KEYS = ("action", "component", "old_text", "new_text", "error_codes", "explanation")
REQUIRED_REWRITE_KEYS = ("action", "sections", "explanation")
REQUIRED_SECTION_KEYS = ("action", "section", "new_text", "error_codes", "explanation")
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


def apply_rewrite(
    sections: dict[str, str],
    proposal: dict[str, Any],
    max_prompt_words: int = 240,
) -> dict[str, Any]:
    """Validate one full rewrite proposal (Section 4: rewrite arm).

    The rewrite must supply all four sections, keep the schema, stay within the
    prompt word limit, and change at least one section. A rejected rewrite
    leaves the current prompt unchanged.
    """
    if not isinstance(proposal, dict):
        return _reason("PROPOSAL_NOT_OBJECT")
    missing = [key for key in REQUIRED_REWRITE_KEYS if key not in proposal]
    if missing:
        return _reason("MISSING_PROPOSAL_KEYS", ", ".join(missing))
    action = proposal.get("action")
    if action not in ("rewrite", "no_change"):
        return _reason("UNKNOWN_ACTION", str(action))
    if action == "no_change":
        return {"ok": True, "reason": None, "sections": dict(sections), "no_change": True}

    rewritten = proposal.get("sections")
    if not isinstance(rewritten, dict):
        return _reason("SECTIONS_NOT_OBJECT")
    missing_sections = [name for name in ALL_COMPONENTS if name not in rewritten]
    if missing_sections:
        return _reason("MISSING_SECTIONS", ", ".join(missing_sections))
    unknown = [name for name in rewritten if name not in ALL_COMPONENTS]
    if unknown:
        return _reason("UNKNOWN_SECTIONS", ", ".join(unknown))
    for name in ALL_COMPONENTS:
        if not isinstance(rewritten[name], str) or not rewritten[name].strip():
            return _reason("EMPTY_SECTION", name)

    introduced = [
        f"{name}:{field}"
        for name in ALL_COMPONENTS
        if (field := _mentions_new_field(rewritten[name]))
    ]
    if introduced:
        return _reason("NEW_FIELD_INTRODUCED", ", ".join(introduced))
    if all(rewritten[name] == sections[name] for name in ALL_COMPONENTS):
        return _reason("NO_OP_EDIT")

    result_sections = {name: rewritten[name] for name in ALL_COMPONENTS}
    words = prompt_word_count(result_sections)
    if words > max_prompt_words:
        return _reason("PROMPT_TOO_LONG", f"{words} words")
    return {"ok": True, "reason": None, "sections": result_sections, "no_change": False}


def apply_section_rewrite(
    sections: dict[str, str],
    proposal: dict[str, Any],
    expected_section: str,
    max_new_text_words: int = 60,
    max_prompt_words: int = 240,
) -> dict[str, Any]:
    """Validate one full-section rewrite for the patch-all-sections design."""
    if expected_section not in ALL_COMPONENTS:
        raise ValueError(f"Unknown section: {expected_section}")
    if not isinstance(proposal, dict):
        return _reason("PROPOSAL_NOT_OBJECT")
    missing = [key for key in REQUIRED_SECTION_KEYS if key not in proposal]
    if missing:
        return _reason("MISSING_PROPOSAL_KEYS", ", ".join(missing))
    action = proposal.get("action")
    if action not in ("replace", "no_change"):
        return _reason("UNKNOWN_ACTION", str(action))
    if action == "no_change":
        return {"ok": True, "reason": None, "sections": dict(sections), "no_change": True}

    if proposal.get("section") != expected_section:
        return _reason("SECTION_MISMATCH", f"{proposal.get('section')} != {expected_section}")
    new_text = proposal.get("new_text")
    if not isinstance(new_text, str) or not new_text.strip():
        return _reason("NEW_TEXT_EMPTY")
    if new_text == sections[expected_section]:
        return _reason("NO_OP_EDIT")
    if len(new_text.split()) > max_new_text_words:
        return _reason("NEW_TEXT_TOO_LONG", f"{len(new_text.split())} words")
    new_field = _mentions_new_field(new_text)
    if new_field:
        return _reason("NEW_FIELD_INTRODUCED", new_field)

    patched = dict(sections)
    patched[expected_section] = new_text
    words = prompt_word_count(patched)
    if words > max_prompt_words:
        return _reason("PROMPT_TOO_LONG", f"{words} words")
    return {"ok": True, "reason": None, "sections": patched, "no_change": False}
