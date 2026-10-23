import pytest

from src.localizers import (
    TIE_BREAK_ORDER,
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


def test_llm_localizer_is_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        llm_localizer({}, {})
