import json

from src.storage import DATA_DIR, check_family_isolation, load_subset


def test_split_manifest_matches_planned_blueprint():
    manifest = json.loads((DATA_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "planned"
    planned = manifest["planned_counts"]
    assert {s: sum(c[s] for c in planned.values()) for s in ("feedback", "selection", "test")} == {
        "feedback": 12,
        "selection": 8,
        "test": 40,
    }
    assert all(sum(c.values()) == c["feedback"] + c["selection"] + c["test"] for c in planned.values())


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
