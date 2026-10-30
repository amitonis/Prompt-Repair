import json

from src.evaluator import score_response
from src.storage import DATA_DIR, check_family_isolation, load_jsonl, load_subset


def _manifest():
    return json.loads((DATA_DIR / "split_manifest.json").read_text(encoding="utf-8"))


def _all_records():
    return (
        load_jsonl("feedback.jsonl")
        + load_jsonl("selection.jsonl")
        + load_jsonl("test_inputs.jsonl")
    )


def _with_gold(record):
    if record["split"] != "test":
        return record
    labels = {r["id"]: r for r in load_jsonl("test_labels.jsonl")}
    return {**record, "gold": labels[record["id"]]["gold"]}


def test_split_manifest_matches_planned_blueprint():
    manifest = _manifest()
    assert manifest["status"] in {"planned", "review_pending", "frozen"}
    planned = manifest["planned_counts"]
    assert {s: sum(c[s] for c in planned.values()) for s in ("feedback", "selection", "test")} == {
        "feedback": 10,
        "selection": 10,
        "test": 40,
    }
    assert all(sum(c.values()) == c["feedback"] + c["selection"] + c["test"] for c in planned.values())


def test_record_counts_match_planned_counts():
    records = _all_records()
    assert len(records) == 60
    assert len({r["id"] for r in records}) == 60
    for category, counts in _manifest()["planned_counts"].items():
        for split in ("feedback", "selection", "test"):
            actual = sum(1 for r in records if r["category"] == category and r["split"] == split)
            assert actual == counts[split], f"{category}/{split}: {actual} != {counts[split]}"


def test_families_do_not_cross_splits():
    assert check_family_isolation(_all_records()) == []


def test_manifest_records_match_data_files():
    manifest_records = {r["id"]: r for r in _manifest()["records"]}
    assert set(manifest_records) == {r["id"] for r in _all_records()}
    for record in _all_records():
        listed = manifest_records[record["id"]]
        assert listed["family_id"] == record["family_id"]
        assert listed["split"] == record["split"]
        assert listed["category"] == record["category"]


def test_test_labels_cover_all_test_inputs():
    inputs = load_jsonl("test_inputs.jsonl")
    labels = {r["id"] for r in load_jsonl("test_labels.jsonl")}
    assert labels == {r["id"] for r in inputs}
    assert len(inputs) == 40


def test_load_subset_merges_test_labels():
    merged = load_subset("test")
    assert len(merged) == 40
    assert all("gold" in r for r in merged)


def test_every_gold_is_schema_valid():
    for record in _all_records():
        gold = _with_gold(record)["gold"]
        assert score_response(json.dumps(gold), gold)["record_correct"], record["id"]


def test_non_null_gold_appears_verbatim_in_notice():
    for record in _all_records():
        merged = _with_gold(record)
        notice = " ".join(merged["notice"].split())
        for field, value in merged["gold"].items():
            if value is not None:
                assert " ".join(value.split()) in notice, f"{record['id']}: {field}"


def test_family_isolation_detects_cross_split_family():
    records = [
        {"id": "A1", "family_id": "fam", "split": "feedback", "notice": "x"},
        {"id": "A2", "family_id": "fam", "split": "test", "notice": "x"},
        {"id": "B1", "family_id": "fam2", "split": "test", "notice": "x"},
    ]
    assert check_family_isolation(records) == ["fam"]
    assert check_family_isolation(records[:1] + records[2:]) == []


def test_unknown_split_is_rejected():
    try:
        load_subset("final_test")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown split must raise ValueError")
