import json

import pytest

from src.optimizer import (
    choose_candidate,
    parse_editor_json,
    render_editor_prompt,
    run_rewrite_branch,
    run_span_edit_branch,
    run_targeted_branch,
    should_accept,
)

BASE = {
    "instruction": "Extract details about the target college event from the supplied notice.",
    "context": "The notice describes one clearly named target college event.",
    "constraints": "Keep the result concise and use the notice.",
    "output_format": "Return JSON with the keys event, date, and venue.",
}

CURRENT = {"selection_records_correct": 5, "selection_invalid": 0}
CONTRACT = "Extract event, date, and venue."
PACKET = {"error_counts": {"DATE_MISMATCH": 2}, "failed_examples": []}

METRICS = {
    "v1": {"selection_records_correct": 7, "selection_invalid": 0},
    "v2": {"selection_records_correct": 8, "selection_invalid": 0},
    "span": {"selection_records_correct": 7, "selection_invalid": 0},
    "bad": {"selection_records_correct": 5, "selection_invalid": 0},
    "invalid": {"selection_records_correct": 9, "selection_invalid": 2},
}


class Scripted:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def _sections(marker):
    return {**BASE, "constraints": marker}


def _rewrite_json(marker, action="rewrite"):
    return json.dumps(
        {"action": action, "sections": _sections(marker), "explanation": "evidence"}
    )


def _span_json(new_text="span", component="constraints", old_text=BASE["constraints"]):
    return json.dumps(
        {
            "action": "replace",
            "component": component,
            "old_text": old_text,
            "new_text": new_text,
            "error_codes": ["DATE_MISMATCH"],
            "explanation": "evidence",
        }
    )


def _score(sections):
    return METRICS[sections["constraints"]]


def _run_rewrite(responses, *, max_attempts=2):
    return run_rewrite_branch(
        BASE,
        CURRENT,
        ask=Scripted(responses),
        score=_score,
        task_contract=CONTRACT,
        feedback_packet=PACKET,
        max_attempts=max_attempts,
    )


def test_rewrite_picks_the_highest_accuracy_across_attempts():
    result = _run_rewrite([_rewrite_json("v1"), _rewrite_json("v2")])
    assert result.accepted is True
    assert result.stopped_reason == "accepted"
    assert result.sections["constraints"] == "v2"
    assert result.metrics == METRICS["v2"]
    assert len(result.attempts) == 2
    assert result.attempts[0]["metrics"] == METRICS["v1"]


def test_rewrite_ties_go_to_the_earlier_attempt():
    result = _run_rewrite([_rewrite_json("v1"), _rewrite_json("v1")])
    assert result.accepted is True
    assert result.attempts[0]["accepted"] is True
    assert result.attempts[1]["accepted"] is True
    assert result.sections["constraints"] == "v1"


def test_rewrite_without_improvement_leaves_the_prompt_unchanged():
    result = _run_rewrite([_rewrite_json("bad"), _rewrite_json("bad")])
    assert result.accepted is False
    assert result.sections == BASE
    assert result.metrics == CURRENT
    assert result.stopped_reason == "no_candidate_accepted"


def test_no_change_stops_the_branch():
    adapter = Scripted([_rewrite_json(None, action="no_change")])
    result = run_rewrite_branch(
        BASE,
        CURRENT,
        ask=adapter,
        score=_score,
        task_contract=CONTRACT,
        feedback_packet=PACKET,
    )
    assert result.accepted is False
    assert result.stopped_reason == "no_change"
    assert len(result.attempts) == 1
    assert len(adapter.prompts) == 1


def test_malformed_response_then_valid_response():
    result = _run_rewrite(["not json at all", _rewrite_json("v1")])
    assert result.accepted is True
    assert result.attempts[0]["valid"] is False
    assert result.attempts[0]["reason"].startswith("PARSE_ERROR")
    assert result.attempts[1]["valid"] is True


def test_rewrite_prompt_contains_contract_sections_and_feedback():
    adapter = Scripted([_rewrite_json(None, action="no_change")])
    run_rewrite_branch(
        BASE,
        CURRENT,
        ask=adapter,
        score=_score,
        task_contract=CONTRACT,
        feedback_packet=PACKET,
    )
    prompt = adapter.prompts[0]
    assert CONTRACT in prompt
    assert '"constraints"' in prompt
    assert "DATE_MISMATCH" in prompt
    assert "{{" not in prompt
    assert "}}" not in prompt


def test_span_edit_applies_a_permitted_component():
    result = run_span_edit_branch(
        BASE,
        CURRENT,
        ask=Scripted([_span_json("span")]),
        score=_score,
        task_contract=CONTRACT,
        feedback_packet=PACKET,
        max_attempts=1,
    )
    assert result.accepted is True
    assert result.sections["constraints"] == "span"
    assert result.sections["instruction"] == BASE["instruction"]
    assert result.stopped_reason == "accepted"


def test_span_edit_rejects_a_disallowed_component():
    response = _span_json("span", component="instruction", old_text=BASE["instruction"])
    result = run_span_edit_branch(
        BASE,
        CURRENT,
        ask=Scripted([response]),
        score=_score,
        task_contract=CONTRACT,
        feedback_packet=PACKET,
        max_attempts=1,
    )
    assert result.accepted is False
    assert result.attempts[0]["reason"] == "COMPONENT_NOT_ALLOWED"
    assert result.sections == BASE


def test_span_edit_retries_after_a_rejected_patch():
    rejected = _span_json("span", component="instruction", old_text=BASE["instruction"])
    result = run_span_edit_branch(
        BASE,
        CURRENT,
        ask=Scripted([rejected, _span_json("span")]),
        score=_score,
        task_contract=CONTRACT,
        feedback_packet=PACKET,
    )
    assert result.accepted is True
    assert result.attempts[0]["valid"] is False
    assert result.attempts[1]["valid"] is True


def test_acceptance_requires_strict_gain_and_no_new_invalidity():
    assert should_accept(CURRENT, METRICS["v1"]) is True
    assert should_accept(CURRENT, METRICS["bad"]) is False
    assert should_accept(CURRENT, METRICS["invalid"]) is False
    assert should_accept(CURRENT, {**CURRENT, "selection_invalid": -1, "selection_records_correct": 6}) is True


def test_choose_candidate_prefers_the_first_of_equal_metrics():
    first = {"selection_records_correct": 7, "selection_invalid": 0, "tag": "first"}
    second = {"selection_records_correct": 7, "selection_invalid": 0, "tag": "second"}
    chosen = choose_candidate(CURRENT, [first, second])
    assert chosen["tag"] == "first"


def test_parse_editor_json_tolerates_fences_and_surrounding_text():
    assert parse_editor_json('```json\n{"action": "no_change"}\n```') == {"action": "no_change"}
    assert parse_editor_json('Here it is: {"action": "no_change"} Done.') == {"action": "no_change"}


@pytest.mark.parametrize("raw", ["", "   ", "no braces", "{bad json", "[1, 2]"])
def test_parse_editor_json_rejects_invalid_payloads(raw):
    with pytest.raises(ValueError):
        parse_editor_json(raw)


def test_render_editor_prompt_rejects_unknown_templates():
    with pytest.raises(ValueError):
        render_editor_prompt(
            "unknown",
            task_contract=CONTRACT,
            sections=BASE,
            feedback_packet=PACKET,
        )


SECTION_METRICS = {
    "MARK_INSTR": {"selection_records_correct": 4, "selection_invalid": 0},
    "MARK_CTX": {"selection_records_correct": 4, "selection_invalid": 0},
    "MARK_CON": {"selection_records_correct": 7, "selection_invalid": 0},
    "MARK_OUT": {"selection_records_correct": 4, "selection_invalid": 0},
    "BETTER": {"selection_records_correct": 9, "selection_invalid": 0},
}


def _section_json(section, text, action="replace"):
    return json.dumps(
        {
            "action": action,
            "section": section if action == "replace" else None,
            "new_text": text if action == "replace" else None,
            "error_codes": [],
            "explanation": "evidence",
        }
    )


def _section_score(sections):
    for value in sections.values():
        if value in SECTION_METRICS:
            return SECTION_METRICS[value]
    return CURRENT


def _run_sections(responses, *, max_attempts=2):
    return run_targeted_branch(
        BASE,
        CURRENT,
        ask=Scripted(responses),
        score=_section_score,
        task_contract=CONTRACT,
        feedback_packet=PACKET,
        max_attempts=max_attempts,
    )


def test_section_patches_are_generated_for_all_four_sections():
    responses = [
        _section_json("instruction", "MARK_INSTR"),
        _section_json("context", "MARK_CTX"),
        _section_json("constraints", "MARK_CON"),
        _section_json("output_format", "MARK_OUT"),
    ]
    result = _run_sections(responses)
    assert set(result.patches) == {"instruction", "context", "constraints", "output_format"}
    assert result.accepted is True
    assert result.accepted_section == "constraints"
    assert result.sections["constraints"] == "MARK_CON"
    assert result.patches["constraints"]["accepted"] is True
    assert result.patches["instruction"]["accepted"] is False
    assert result.patches["instruction"]["metrics"] == SECTION_METRICS["MARK_INSTR"]
    assert len(result.attempts) == 1


def test_section_patches_retry_only_when_nothing_is_accepted():
    low = {"selection_records_correct": 4, "selection_invalid": 0}
    first = [
        _section_json(name, f"LOW_{name}")
        for name in ("instruction", "context", "constraints", "output_format")
    ]
    second = [
        _section_json("instruction", "LOW_instruction"),
        _section_json("context", "LOW_context"),
        _section_json("constraints", "BETTER"),
        _section_json("output_format", "LOW_output_format"),
    ]
    score_table = dict(SECTION_METRICS, **{f"LOW_{n}": low for n in ("instruction", "context", "constraints", "output_format")})

    def score(sections):
        for value in sections.values():
            if value in score_table:
                return score_table[value]
        return CURRENT

    result = run_targeted_branch(
        BASE,
        CURRENT,
        ask=Scripted(first + second),
        score=score,
        task_contract=CONTRACT,
        feedback_packet=PACKET,
    )
    assert len(result.attempts) == 2
    assert result.accepted is True
    assert result.accepted_section == "constraints"
    assert result.sections["constraints"] == "BETTER"


def test_all_sections_declining_stops_the_branch():
    responses = [
        _section_json(name, None, action="no_change")
        for name in ("instruction", "context", "constraints", "output_format")
    ]
    result = _run_sections(responses)
    assert result.accepted is False
    assert result.stopped_reason == "no_change"
    assert len(result.attempts) == 1
    assert result.patches["constraints"]["reason"] == "NO_CHANGE"


def test_section_prompts_name_the_target_section():
    adapter = Scripted(
        [
            _section_json(name, None, action="no_change")
            for name in ("instruction", "context", "constraints", "output_format")
        ]
    )
    run_targeted_branch(
        BASE,
        CURRENT,
        ask=adapter,
        score=_section_score,
        task_contract=CONTRACT,
        feedback_packet=PACKET,
    )
    for prompt, name in zip(
        adapter.prompts, ("instruction", "context", "constraints", "output_format")
    ):
        assert f"TARGET SECTION:\n{name}" in prompt
        assert str(PACKET["error_counts"]["DATE_MISMATCH"]) in prompt or "DATE_MISMATCH" in prompt


def test_section_patch_with_the_wrong_section_is_rejected():
    result = run_targeted_branch(
        BASE,
        CURRENT,
        ask=Scripted([_section_json("context", "MARK_CTX")]),
        score=_section_score,
        task_contract=CONTRACT,
        feedback_packet=PACKET,
        target_sections=("instruction",),
        max_attempts=1,
    )
    assert result.accepted is False
    assert result.patches["instruction"]["reason"] == "SECTION_MISMATCH"
