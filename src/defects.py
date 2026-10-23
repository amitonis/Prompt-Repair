"""Oracle prompt and single-section defect injection (Sections 9A.2-9A.3)."""

from __future__ import annotations

import json
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
SECTIONS = ("instruction", "context", "constraints", "output_format")


def load_prompt(name: str) -> dict[str, str]:
    path = PROMPTS_DIR / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    sections = data["sections"]
    missing = [s for s in SECTIONS if s not in sections]
    if missing:
        raise ValueError(f"Prompt {name} is missing sections: {missing}")
    return {s: sections[s] for s in SECTIONS}


def load_defects() -> list[dict]:
    defects_dir = PROMPTS_DIR / "defects"
    defects = []
    for path in sorted(defects_dir.glob("D*.json")):
        defect = json.loads(path.read_text(encoding="utf-8"))
        for key in ("id", "section", "defect", "expected_symptom", "defective_section_text"):
            if key not in defect:
                raise ValueError(f"Defect file {path.name} is missing key: {key}")
        if defect["section"] not in SECTIONS:
            raise ValueError(f"Defect {defect['id']} names unknown section: {defect['section']}")
        defects.append(defect)
    return defects


def inject_defect(oracle_sections: dict[str, str], defect: dict) -> dict[str, str]:
    """Apply one defect by replacing exactly one section of the oracle prompt."""
    section = defect["section"]
    if section not in oracle_sections:
        raise ValueError(f"Unknown section: {section}")
    replacement = defect["defective_section_text"]
    if replacement == oracle_sections[section]:
        raise ValueError(f"Defect {defect['id']} does not change the section")
    defective = dict(oracle_sections)
    defective[section] = replacement
    return defective


def changed_sections(left: dict[str, str], right: dict[str, str]) -> list[str]:
    return [s for s in SECTIONS if left[s] != right[s]]
