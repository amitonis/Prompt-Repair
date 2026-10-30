import json

import pytest

from src.experiment_runner import (
    MANIFEST_FIELDS,
    append_cache,
    cache_key,
    has_feedback_signal,
    load_cache,
    manifest_diff,
    missing_jobs,
    oracle_ceiling_met,
    original_prompt_has_headroom,
    run_generations,
    screen_defects,
    session_manifest,
    write_session_manifest,
)
from src.inference import GenerationRecord


def test_cache_key_is_deterministic():
    a = cache_key("model-x", "rev1", "vllm 0.6", None, "prompt text", "notice text", 42)
    b = cache_key("model-x", "rev1", "vllm 0.6", None, "prompt text", "notice text", 42)
    assert a == b


def test_cache_key_changes_with_each_component():
    base = cache_key("model-x", "rev1", "vllm 0.6", None, "prompt", "notice", 42)
    assert cache_key("model-y", "rev1", "vllm 0.6", None, "prompt", "notice", 42) != base
    assert cache_key("model-x", "rev2", "vllm 0.6", None, "prompt", "notice", 42) != base
    assert cache_key("model-x", "rev1", "vllm 0.7", None, "prompt", "notice", 42) != base
    assert cache_key("model-x", "rev1", "vllm 0.6", "8bit", "prompt", "notice", 42) != base
    assert cache_key("model-x", "rev1", "vllm 0.6", None, "prompt2", "notice", 42) != base
    assert cache_key("model-x", "rev1", "vllm 0.6", None, "prompt", "notice2", 42) != base
    assert cache_key("model-x", "rev1", "vllm 0.6", None, "prompt", "notice", 43) != base


def _manifest(**overrides):
    values = {
        "git_commit": "abc1234",
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "model_revision": "deadbeef",
        "serving_library": "transformers",
        "serving_library_version": "4.99.0",
        "torch_version": "2.9.0",
        "dtype": "bfloat16",
        "quantization": "none",
        "gpu_name": "NVIDIA RTX PRO 6000 Blackwell",
        "cuda_version": "13.0",
        "driver_version": "580.178.04",
        "batch_size": 16,
        "seed": 20261014,
        "started_at_utc": "2026-10-25T10:00:00+00:00",
    }
    values.update(overrides)
    return session_manifest(**values)


def test_session_manifest_has_every_field():
    manifest = _manifest(started_at_utc=None)
    assert tuple(manifest) == MANIFEST_FIELDS
    assert manifest["started_at_utc"].endswith("+00:00")
    assert manifest["batch_size"] == 16


def test_session_manifest_rejects_bad_inputs():
    with pytest.raises(ValueError):
        _manifest(git_commit="")
    with pytest.raises(ValueError):
        _manifest(model_id="")
    with pytest.raises(ValueError):
        _manifest(batch_size=0)


def test_manifest_diff_ignores_timestamp_and_reports_changes():
    first = _manifest()
    second = _manifest(started_at_utc="2026-10-26T09:00:00+00:00", batch_size=8)
    changes = manifest_diff(first, second)
    assert changes == [{"field": "batch_size", "previous": 16, "current": 8}]


def test_write_session_manifest_round_trip(tmp_path):
    target = tmp_path / "nested" / "run_manifest.json"
    manifest = _manifest()
    write_session_manifest(target, manifest)
    assert json.loads(target.read_text(encoding="utf-8")) == manifest
    with pytest.raises(ValueError):
        write_session_manifest(target, {"git_commit": "abc"})


def test_cache_is_append_only_and_first_record_wins(tmp_path):
    cache = tmp_path / "cache.jsonl"
    assert load_cache(cache) == {}
    assert append_cache(cache, {"cache_key": "k1", "raw": "first"})
    assert append_cache(cache, {"cache_key": "k2", "raw": "second"})
    assert not append_cache(cache, {"cache_key": "k1", "raw": "overwrite"})
    records = load_cache(cache)
    assert records["k1"]["raw"] == "first"
    assert records["k2"]["raw"] == "second"
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 2


def test_missing_jobs_preserves_order_and_filters_cached(tmp_path):
    cache = tmp_path / "cache.jsonl"
    append_cache(cache, {"cache_key": "b", "raw": "done"})
    jobs = [("a", 1), ("b", 2), ("c", 3)]
    assert missing_jobs(cache, jobs) == [("a", 1), ("c", 3)]


DEV_IDS = [f"N{i:02d}" for i in range(1, 21)]


def _dev_scores(failing=()):
    failing = set(failing)
    return {n: n not in failing for n in DEV_IDS}


def test_standard_screening_keeps_a_15_point_drop():
    oracle = _dev_scores()
    outcomes = screen_defects(
        oracle,
        {"D1": _dev_scores({"N01", "N02", "N03"})},
        target_notices={"D1": ["N01", "N02", "N03", "N04"]},
    )
    result = outcomes[0]
    assert result.kept is True
    assert result.drop_points == 15.0
    assert result.reason == "standard 15-point rule"


def test_standard_screening_drops_a_10_point_drop():
    outcomes = screen_defects(
        _dev_scores(),
        {"D1": _dev_scores({"N01", "N02"})},
        target_notices={"D1": ["N01", "N02"]},
        thin_defect_ids=(),
    )
    assert outcomes[0].kept is False
    assert outcomes[0].drop_points == 10.0


def test_d2_requires_three_of_four_targeted_flips():
    oracle = _dev_scores()
    incidental = screen_defects(
        oracle,
        {"D2": _dev_scores({"N05", "N06", "N07"})},
        target_notices={"D2": ["N01", "N02", "N03", "N04"]},
    )[0]
    assert incidental.kept is False
    assert incidental.reason == "flip rule (0/4, needs 3)"

    genuine = screen_defects(
        oracle,
        {"D2": _dev_scores({"N01", "N02", "N03"})},
        target_notices={"D2": ["N01", "N02", "N03", "N04"]},
    )[0]
    assert genuine.kept is True
    assert genuine.target_size == 4


def test_d3_thin_category_needs_every_targeted_notice_to_flip():
    oracle = _dev_scores()
    kept = screen_defects(
        oracle,
        {"D3": _dev_scores({"N01", "N02"})},
        target_notices={"D3": ["N01", "N02"]},
    )[0]
    assert kept.kept is True
    assert kept.drop_points == 10.0
    assert kept.reason == "thin-category rule"

    partial = screen_defects(
        oracle,
        {"D3": _dev_scores({"N01"})},
        target_notices={"D3": ["N01", "N02"]},
    )[0]
    assert partial.kept is False
    assert partial.flips == ("N01",)


def test_screening_rejects_inconsistent_inputs():
    with pytest.raises(ValueError):
        screen_defects({}, {}, target_notices={})
    with pytest.raises(ValueError):
        screen_defects(
            _dev_scores(),
            {"D1": {"N01": False}},
            target_notices={"D1": ["N01"]},
        )
    with pytest.raises(ValueError):
        screen_defects(
            _dev_scores(),
            {"D1": _dev_scores()},
            target_notices={"D1": ["N99"]},
        )
    with pytest.raises(ValueError):
        screen_defects(
            _dev_scores(),
            {"D1": _dev_scores()},
            target_notices={"D1": []},
        )


def test_ceiling_headroom_and_feedback_signal():
    assert oracle_ceiling_met(_dev_scores({"N01", "N02", "N03"})) is True
    assert oracle_ceiling_met(_dev_scores({"N01", "N02", "N03", "N04"})) is False
    assert original_prompt_has_headroom(_dev_scores({"N01", "N02", "N03"})) is True
    assert original_prompt_has_headroom(_dev_scores({"N01", "N02"})) is False
    assert has_feedback_signal(_dev_scores({"N01"}), ["N01", "N02"]) is True
    assert has_feedback_signal(_dev_scores(), ["N01", "N02"]) is False
    with pytest.raises(ValueError):
        has_feedback_signal(_dev_scores(), [])


class _StubAdapter:
    model_id = "fake/model"
    revision = "rev123"
    quantization = "none"
    serving_library_version = "4.99.0"

    def __init__(self, fail=False):
        self.fail = fail
        self.prompts = []

    def generate_batch(self, prompts, **settings):
        self.prompts.extend(prompts)
        return [
            GenerationRecord(
                run_id=settings.get("run_id", ""),
                prompt_hash=f"h{index}",
                model_id=self.model_id,
                model_revision=self.revision,
                raw="" if self.fail else f"out:{prompt}",
                status="error" if self.fail else "ok",
                duration_s=0.01,
                usage={"error": "boom"} if self.fail else {},
                serving_library="transformers",
                quantization=self.quantization,
                seed=settings.get("seed"),
            )
            for index, prompt in enumerate(prompts)
        ]


def _jobs(count=2):
    return [
        {
            "setting_id": "D1",
            "arm": "rewrite",
            "notice_id": f"F{index}",
            "prompt": f"p{index}",
            "notice": f"n{index}",
        }
        for index in range(count)
    ]


def test_run_generations_caches_successes(tmp_path):
    cache = tmp_path / "cache.jsonl"
    adapter = _StubAdapter()
    results = run_generations(adapter, _jobs(), cache, run_id="run-1", seed=1)
    assert [r["raw"] for r in results] == ["out:p0", "out:p1"]
    assert all(r["cached"] is False for r in results)
    assert results[0]["setting_id"] == "D1"
    assert results[0]["run_id"] == "run-1"
    assert results[0]["seed"] == 1
    assert len(load_cache(cache)) == 2


def test_run_generations_resumes_from_the_cache(tmp_path):
    cache = tmp_path / "cache.jsonl"
    run_generations(_StubAdapter(), _jobs(), cache, run_id="run-1", seed=1)
    second = _StubAdapter()
    results = run_generations(second, _jobs(), cache, run_id="run-2", seed=1)
    assert second.prompts == []
    assert all(r["cached"] is True for r in results)
    assert results[0]["run_id"] == "run-1"
    assert len(load_cache(cache)) == 2


def test_run_generations_retries_failures_and_does_not_cache_them(tmp_path):
    cache = tmp_path / "cache.jsonl"
    failed = run_generations(_StubAdapter(fail=True), _jobs(), cache, run_id="run-1")
    assert [r["status"] for r in failed] == ["error", "error"]
    assert load_cache(cache) == {}

    healthy = _StubAdapter()
    retried = run_generations(healthy, _jobs(), cache, run_id="run-1")
    assert [r["status"] for r in retried] == ["ok", "ok"]
    assert healthy.prompts == ["p0", "p1"]
    assert len(load_cache(cache)) == 2


def test_run_generations_deduplicates_identical_requests(tmp_path):
    cache = tmp_path / "cache.jsonl"
    jobs = _jobs(2)
    jobs[1]["prompt"] = jobs[0]["prompt"]
    jobs[1]["notice"] = jobs[0]["notice"]
    adapter = _StubAdapter()
    results = run_generations(adapter, jobs, cache, run_id="run-1")
    assert adapter.prompts == ["p0"]
    assert [r["raw"] for r in results] == ["out:p0", "out:p0"]
    assert len(load_cache(cache)) == 1


def test_run_generations_mixes_cached_and_new_in_job_order(tmp_path):
    cache = tmp_path / "cache.jsonl"
    jobs = _jobs(2)
    run_generations(_StubAdapter(), jobs[:1], cache, run_id="run-1")
    adapter = _StubAdapter()
    results = run_generations(adapter, jobs, cache, run_id="run-2")
    assert adapter.prompts == ["p1"]
    assert [r["cached"] for r in results] == [True, False]
    assert [r["notice_id"] for r in results] == ["F0", "F1"]


def test_run_generations_requires_job_fields(tmp_path):
    with pytest.raises(ValueError):
        run_generations(_StubAdapter(), [{"prompt": "p"}], tmp_path / "c.jsonl", run_id="r")


def test_invalid_cache_records_are_rejected(tmp_path):
    cache = tmp_path / "cache.jsonl"
    cache.write_text('{"raw": "no key"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_cache(cache)
    with pytest.raises(ValueError):
        append_cache(cache, {"raw": "no key"})
    with pytest.raises(ValueError):
        missing_jobs(cache, [("k", 1)])
