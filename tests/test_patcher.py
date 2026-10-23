import pytest

from src.patcher import apply_patch

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
