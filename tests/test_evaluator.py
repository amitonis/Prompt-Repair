import json
import pytest

from src.evaluator import levenshtein_chars, score_response

GOLD = {
    "event": "AI Workshop",
    "date": "20 October 2026",
    "venue": None,
}


def test_correct_response():
    score = score_response(json.dumps(GOLD), GOLD)
    assert score["record_correct"]
    assert score["schema_valid"]
    assert sum(score["field_correct"].values()) == 3


def test_missing_key_is_not_null():
    raw = json.dumps({"event": "AI Workshop", "date": "20 October 2026"})
    score = score_response(raw, GOLD)
    assert not score["record_correct"]
    assert not score["field_correct"]["venue"]
    assert "SCHEMA_KEYS" in score["issues"]


def test_extra_key_fails_record_but_preserves_field_diagnostics():
    score = score_response(json.dumps({**GOLD, "organizer": "AI Club"}), GOLD)
    assert score["json_valid"]
    assert not score["schema_valid"]
    assert not score["record_correct"]
    assert all(score["field_correct"].values())


@pytest.mark.parametrize("venue", ["null", "Main Hall", "", 0, False])
def test_non_null_value_is_not_null(venue):
    score = score_response(json.dumps({**GOLD, "venue": venue}), GOLD)
    assert not score["record_correct"]
    assert not score["field_correct"]["venue"]


@pytest.mark.parametrize("raw", [
    'Here is the result: {"event":"AI Workshop","date":null,"venue":null}',
    '```json\n{"event":"AI Workshop","date":null,"venue":null}\n```',
    '{"event":"AI Workshop","date":null,"venue":null,"venue":"Hall A"}',
    '{"event":"AI Workshop","date":NaN,"venue":null}',
    '',
])
def test_strict_parsing(raw):
    score = score_response(raw, GOLD)
    assert not score["json_valid"]
    assert not score["record_correct"]


def test_top_level_array_is_json_but_not_our_schema():
    score = score_response(json.dumps([GOLD]), GOLD)
    assert score["json_valid"]
    assert not score["schema_valid"]


def test_wrong_date_is_semantic_error():
    score = score_response(json.dumps({**GOLD, "date": "18 October 2026"}), GOLD)
    assert score["schema_valid"]
    assert not score["record_correct"]
    assert "DATE_MISMATCH" in score["issues"]


def test_whitespace_normalization_only():
    score = score_response(json.dumps({**GOLD, "event": " AI   Workshop "}), GOLD)
    assert score["record_correct"]
    changed_case = score_response(json.dumps({**GOLD, "event": "ai workshop"}), GOLD)
    assert not changed_case["record_correct"]


def test_invalid_gold_is_rejected():
    with pytest.raises(ValueError):
        score_response("{}", {"event": "AI Workshop"})


def test_edit_distance():
    assert levenshtein_chars("", "abc") == 3
    assert levenshtein_chars("prompt", "prompt") == 0
    assert levenshtein_chars("kitten", "sitting") == 3
