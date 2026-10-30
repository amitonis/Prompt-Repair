import pytest

from src.defects import load_prompt
from src.localizers import (
    TIE_BREAK_ORDER,
    behav_localizer_rank,
    behav_neutralize,
    llm_localizer,
    rule_localizer,
)


def test_rule_localizer_prefers_constraints_for_null_policy_violations():
    packet = {"error_counts": {"NULL_POLICY_VIOLATION": 3, "DATE_MISMATCH": 1}}
    assert rule_localizer(packet)[0] == "constraints"


def test_rule_localizer_prefers_output_format_for_malformed_json():
    packet = {"error_counts": {"MALFORMED_JSON": 2}}
    assert rule_localizer(packet)[0] == "output_format"


def test_rule_localizer_uses_fixed_tie_break_order():
    assert rule_localizer({"error_counts": {}}) == list(TIE_BREAK_ORDER)


def test_rule_localizer_weights_by_error_counts():
    packet = {"error_counts": {"NULL_POLICY_VIOLATION": 1, "MALFORMED_JSON": 5}}
    assert rule_localizer(packet)[0] == "output_format"


def test_behav_localizer_ranks_by_output_change_share():
    changes = {
        "instruction": [True, True],
        "output_format": [True, True],
        "constraints": [True, False],
        "context": [False, False],
    }
    ranking = behav_localizer_rank(changes)
    assert ranking[0] == "output_format"
    assert ranking[1] == "instruction"
    assert ranking[2] == "constraints"
    assert ranking[3] == "context"


def test_behav_localizer_handles_empty_section():
    ranking = behav_localizer_rank({})
    assert ranking == list(TIE_BREAK_ORDER)


def test_behav_neutralize_uses_original_section():
    oracle = load_prompt("oracle")
    original = load_prompt("original")
    neutralized = behav_neutralize(oracle, "constraints")
    assert neutralized["constraints"] == original["constraints"]
    for section in ("instruction", "context", "output_format"):
        assert neutralized[section] == oracle[section]


def test_behav_neutralize_rejects_unknown_section():
    with pytest.raises(ValueError):
        behav_neutralize(load_prompt("oracle"), "notice")


def test_llm_localizer_is_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        llm_localizer({}, {})
