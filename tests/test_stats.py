from src.stats import paired_cluster_bootstrap


def _records(family_id: str, correct_a: bool, correct_b: bool, count: int):
    return [(family_id, correct_a, correct_b)] * count


def test_identical_arms_have_zero_interval():
    result = paired_cluster_bootstrap(_records("fam", True, True, 4))
    assert result["observed_delta"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0
    assert result["p_value"] == 1.0


def test_perfectly_separated_arms_have_unit_interval():
    result = paired_cluster_bootstrap(_records("fam", True, False, 4))
    assert result["observed_delta"] == 1.0
    assert result["ci_low"] == 1.0
    assert result["ci_high"] == 1.0
    assert result["p_value"] == 0.0


def test_two_family_distribution_spans_zero_and_one():
    records = _records("family_a", True, False, 2) + _records("family_b", True, True, 2)
    result = paired_cluster_bootstrap(records, n_resamples=2000, seed=7)
    assert result["n_families"] == 2
    assert result["n_notices"] == 4
    assert result["observed_delta"] == 0.5
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 1.0


def test_single_family_is_fully_carried_every_resample():
    records = [
        ("fam", True, False),
        ("fam", False, True),
        ("fam", False, False),
        ("fam", True, True),
    ]
    result = paired_cluster_bootstrap(records, n_resamples=500, seed=11)
    assert result["n_families"] == 1
    assert result["observed_delta"] == 0.0
    assert result["ci_low"] == result["ci_high"] == 0.0


def test_fixed_seed_is_deterministic():
    records = (
        _records("a", True, False, 3)
        + _records("b", True, True, 2)
        + _records("c", False, True, 3)
        + _records("d", True, False, 1)
    )
    first = paired_cluster_bootstrap(records, n_resamples=1000, seed=20261014)
    second = paired_cluster_bootstrap(records, n_resamples=1000, seed=20261014)
    assert first == second
    other = paired_cluster_bootstrap(records, n_resamples=1000, seed=1)
    assert other["observed_delta"] == first["observed_delta"]
    assert other["n_families"] == first["n_families"] == 4
    assert other["n_notices"] == first["n_notices"] == 9


def test_confidence_interval_brackets_observed_delta():
    records = (
        _records("a", True, False, 2)
        + _records("b", True, True, 2)
        + _records("c", True, False, 1)
    )
    result = paired_cluster_bootstrap(records, n_resamples=3000, seed=3)
    assert result["ci_low"] <= result["observed_delta"] <= result["ci_high"]
    assert result["ci_low"] < result["ci_high"]


def test_invalid_inputs_are_rejected():
    for bad in (
        lambda: paired_cluster_bootstrap([]),
        lambda: paired_cluster_bootstrap([("", True, False)]),
        lambda: paired_cluster_bootstrap([("fam", True, False)], n_resamples=0),
        lambda: paired_cluster_bootstrap([("fam", True, False)], confidence=1.0),
    ):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
