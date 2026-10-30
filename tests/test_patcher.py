import pytest

from src.patcher import apply_patch, apply_rewrite, apply_section_rewrite

BASE = {
    "instruction": "Extract details about the target college event from the supplied notice.",
    "context": "The notice describes one clearly named target college event.",
    "constraints": "Keep the result concise and use the notice.",
    "output_format": "Return JSON with the keys event, date, and venue.",
}

VALID = {
    "action": "replace",
    "component": "constraints",
    "old_text": "Keep the result concise and use the notice.",
    "new_text": "Use only explicitly stated information. Return null for absent or unresolved values.",
    "error_codes": ["NULL_POLICY_VIOLATION"],
    "explanation": "Non-null venue where the notice supplies no location.",
}


def test_valid_edit_alters_exactly_one_permitted_component():
    result = apply_patch(BASE, VALID)
    assert result["ok"]
    for section, text in BASE.items():
        if section != "constraints":
            assert result["sections"][section] == text
    assert result["sections"]["constraints"] == VALID["new_text"]


def test_instruction_edit_rejected_in_restricted_baseline():
    proposal = dict(VALID, component="instruction", old_text=BASE["instruction"], new_text="Extract the event details.")
    result = apply_patch(BASE, proposal)
    assert not result["ok"]
    assert result["reason"] == "COMPONENT_NOT_ALLOWED"


def test_instruction_edit_allowed_when_all_components_eligible():
    proposal = dict(VALID, component="instruction", old_text=BASE["instruction"], new_text="Extract the event name, date, and venue.")
    result = apply_patch(BASE, proposal, allowed_components=("instruction", "context", "constraints", "output_format"))
    assert result["ok"]


def test_old_span_occurring_twice_is_rejected():
    sections = dict(BASE, constraints="Be concise. Be concise about the event.")
    proposal = dict(VALID, old_text="Be concise", new_text="Be brief")
    result = apply_patch(sections, proposal)
    assert not result["ok"]
    assert result["reason"] == "OLD_TEXT_AMBIGUOUS"


def test_absent_old_span_is_rejected():
    result = apply_patch(BASE, dict(VALID, old_text="Not present in the section."))
    assert not result["ok"]
    assert result["reason"] == "OLD_TEXT_ABSENT"


def test_oversized_replacement_is_rejected():
    long_text = " ".join(["word"] * 61)
    result = apply_patch(BASE, dict(VALID, new_text=long_text))
    assert not result["ok"]
    assert result["reason"] == "NEW_TEXT_TOO_LONG"


def test_new_field_introduction_is_rejected():
    result = apply_patch(BASE, dict(VALID, new_text='Add the key "organizer" to the output.'))
    assert not result["ok"]
    assert result["reason"] == "NEW_FIELD_INTRODUCED"


@pytest.mark.parametrize("proposal,reason", [
    ({"action": "delete", "component": "constraints", "old_text": "x", "new_text": "y", "error_codes": [], "explanation": ""}, "UNKNOWN_ACTION"),
    ({"action": "replace", "old_text": "x", "new_text": "y", "error_codes": [], "explanation": ""}, "MISSING_PROPOSAL_KEYS"),
    (dict(VALID, new_text="   "), "NEW_TEXT_EMPTY"),
    (dict(VALID, old_text=""), "OLD_TEXT_EMPTY"),
    (dict(VALID, new_text=VALID["old_text"]), "NO_OP_EDIT"),
    ("not a dict", "PROPOSAL_NOT_OBJECT"),
])
def test_malformed_proposals_are_rejected(proposal, reason):
    result = apply_patch(BASE, proposal)
    assert not result["ok"]
    assert result["reason"] == reason


def test_rejected_patch_leaves_prompt_unchanged():
    result = apply_patch(BASE, dict(VALID, old_text="absent span"))
    assert not result["ok"]
    assert BASE["constraints"] == "Keep the result concise and use the notice."


def test_no_change_action_is_admissible():
    result = apply_patch(BASE, {
        "action": "no_change",
        "component": None,
        "old_text": None,
        "new_text": None,
        "error_codes": [],
        "explanation": "No justified edit exists.",
    })
    assert result["ok"]
    assert result["no_change"]
    assert result["sections"] == BASE


REWRITE_SECTIONS = {
    "instruction": "Extract the event name, date, and venue from the supplied notice.",
    "context": "The notice describes one clearly named target college event.",
    "constraints": "Use only explicitly stated information and return null for absent values.",
    "output_format": "Return one JSON object with keys event, date, and venue.",
}


def _rewrite_proposal(**overrides):
    proposal = {
        "action": "rewrite",
        "sections": dict(REWRITE_SECTIONS),
        "explanation": "Clarified the null policy.",
    }
    proposal.update(overrides)
    return proposal


def test_rewrite_replaces_all_four_sections():
    result = apply_rewrite(BASE, _rewrite_proposal())
    assert result["ok"]
    assert result["sections"] == REWRITE_SECTIONS
    assert result["no_change"] is False


def test_rewrite_no_change_is_admissible():
    result = apply_rewrite(BASE, _rewrite_proposal(action="no_change", sections=None))
    assert result["ok"]
    assert result["no_change"]
    assert result["sections"] == BASE


@pytest.mark.parametrize("proposal,reason", [
    ({"action": "rewrite", "sections": dict(REWRITE_SECTIONS)}, "MISSING_PROPOSAL_KEYS"),
    (_rewrite_proposal(action="replace"), "UNKNOWN_ACTION"),
    (_rewrite_proposal(sections="text"), "SECTIONS_NOT_OBJECT"),
    (_rewrite_proposal(sections={k: v for k, v in REWRITE_SECTIONS.items() if k != "context"}), "MISSING_SECTIONS"),
    (_rewrite_proposal(sections={**REWRITE_SECTIONS, "examples": "One."}), "UNKNOWN_SECTIONS"),
    (_rewrite_proposal(sections={**REWRITE_SECTIONS, "context": "   "}), "EMPTY_SECTION"),
    (_rewrite_proposal(sections=dict(BASE)), "NO_OP_EDIT"),
    (_rewrite_proposal(sections={**REWRITE_SECTIONS, "constraints": 'Add the key "organizer".'}), "NEW_FIELD_INTRODUCED"),
    (_rewrite_proposal(sections={**REWRITE_SECTIONS, "instruction": " ".join(["word"] * 241)}), "PROMPT_TOO_LONG"),
    ("not a dict", "PROPOSAL_NOT_OBJECT"),
])
def test_invalid_rewrites_are_rejected(proposal, reason):
    result = apply_rewrite(BASE, proposal)
    assert not result["ok"]
    assert result["reason"] == reason


def _section_proposal(section="constraints", new_text="Use only stated information.", action="replace"):
    return {
        "action": action,
        "section": section if action == "replace" else None,
        "new_text": new_text if action == "replace" else None,
        "error_codes": [],
        "explanation": "evidence",
    }


def test_section_rewrite_replaces_only_the_named_section():
    result = apply_section_rewrite(BASE, _section_proposal(), "constraints")
    assert result["ok"]
    assert result["sections"]["constraints"] == "Use only stated information."
    for name in ("instruction", "context", "output_format"):
        assert result["sections"][name] == BASE[name]


def test_section_rewrite_no_change_is_admissible():
    result = apply_section_rewrite(BASE, _section_proposal(action="no_change"), "constraints")
    assert result["ok"]
    assert result["no_change"]
    assert result["sections"] == BASE


@pytest.mark.parametrize("proposal,reason", [
    (_section_proposal(section="context"), "SECTION_MISMATCH"),
    (_section_proposal(new_text="   "), "NEW_TEXT_EMPTY"),
    (_section_proposal(new_text=BASE["constraints"]), "NO_OP_EDIT"),
    (_section_proposal(new_text=" ".join(["word"] * 61)), "NEW_TEXT_TOO_LONG"),
    (_section_proposal(new_text='Add the key "organizer".'), "NEW_FIELD_INTRODUCED"),
    ({"action": "replace", "section": "constraints", "new_text": "x"}, "MISSING_PROPOSAL_KEYS"),
    (_section_proposal(action="rewrite"), "UNKNOWN_ACTION"),
    ("not a dict", "PROPOSAL_NOT_OBJECT"),
])
def test_invalid_section_rewrites_are_rejected(proposal, reason):
    result = apply_section_rewrite(BASE, proposal, "constraints")
    assert not result["ok"]
    assert result["reason"] == reason


def test_section_rewrite_requires_a_known_target():
    with pytest.raises(ValueError):
        apply_section_rewrite(BASE, _section_proposal(), "examples")
