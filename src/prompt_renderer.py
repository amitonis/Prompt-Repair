"""Render the reusable four-section prompt template with a runtime notice."""

from __future__ import annotations

SECTION_ORDER = ("instruction", "context", "constraints", "output_format")
SECTION_LABELS = {
    "instruction": "[INSTRUCTION]",
    "context": "[CONTEXT]",
    "constraints": "[CONSTRAINTS]",
    "output_format": "[OUTPUT_FORMAT]",
}
NOTICE_LABEL = "[NOTICE]"
NOTICE_PLACEHOLDER = "{{NOTICE}}"


def word_count(text: str) -> int:
    return len(text.split())


def render_template(sections: dict[str, str]) -> str:
    """Render the reusable template, excluding the runtime notice content."""
    for name in SECTION_ORDER:
        if not isinstance(sections.get(name), str) or not sections[name].strip():
            raise ValueError(f"Missing or empty section: {name}")
    parts = [f"{SECTION_LABELS[name]}\n{sections[name]}" for name in SECTION_ORDER]
    parts.append(f"{NOTICE_LABEL}\n{NOTICE_PLACEHOLDER}")
    return "\n\n".join(parts)


def render_prompt(sections: dict[str, str], notice: str) -> str:
    """Render the full prompt with a concrete notice substituted in."""
    if not isinstance(notice, str) or not notice.strip():
        raise ValueError("notice must be a nonempty string")
    return render_template(sections).replace(NOTICE_PLACEHOLDER, notice)


def prompt_word_count(sections: dict[str, str]) -> int:
    """Whitespace words of the rendered template, excluding section labels."""
    return sum(word_count(sections[name]) for name in SECTION_ORDER)
