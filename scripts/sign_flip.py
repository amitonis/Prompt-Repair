"""Exact sign-flip test over families (analysis_plan.md Section 8).

Descriptive p-value only: the family-cluster bootstrap confidence interval
carries the inference. ``records`` holds one ``(family_id, correct_a,
correct_b)`` triple per notice; family totals are sign-flipped exhaustively
over all 2^k assignments.

Run ``python scripts/sign_flip.py --self-test`` for the known-answer checks.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from itertools import product

MAX_EXACT_FAMILIES = 20
_EPSILON = 1e-12


def exact_sign_flip(records: Iterable[tuple[str, bool, bool]]) -> dict[str, float | int]:
    totals: dict[str, int] = {}
    counts: dict[str, int] = {}
    for family_id, correct_a, correct_b in records:
        if not isinstance(family_id, str) or not family_id:
            raise ValueError("family_id must be a nonempty string")
        totals[family_id] = totals.get(family_id, 0) + (int(bool(correct_a)) - int(bool(correct_b)))
        counts[family_id] = counts.get(family_id, 0) + 1
    if not totals:
        raise ValueError("At least one record is required")
    if len(totals) > MAX_EXACT_FAMILIES:
        raise ValueError(f"exact enumeration is limited to {MAX_EXACT_FAMILIES} families")

    values = list(totals.values())
    n_notices = sum(counts.values())
    observed_sum = sum(values)
    observed = observed_sum / n_notices

    extreme = 0
    for signs in product((1, -1), repeat=len(values)):
        signed_sum = sum(sign * value for sign, value in zip(signs, values))
        if abs(signed_sum) >= abs(observed_sum) - _EPSILON:
            extreme += 1
    n_patterns = 2 ** len(values)

    return {
        "observed_delta": observed,
        "p_value": extreme / n_patterns,
        "n_families": len(values),
        "n_notices": n_notices,
        "n_sign_patterns": n_patterns,
    }


def _self_test() -> int:
    checks: list[tuple[str, object, object]] = []

    identical = [("fam", True, True)] * 4
    checks.append(("identical arms", exact_sign_flip(identical)["p_value"], 1.0))

    separated = [("fam", True, False)] * 4
    checks.append(("single separated family", exact_sign_flip(separated)["p_value"], 1.0))

    four = [(f"f{i}", True, False) for i in range(4) for _ in range(2)]
    checks.append(("four separated families", exact_sign_flip(four)["p_value"], 0.125))

    mixed = [("a", True, False), ("a", True, False), ("b", False, True), ("b", False, True)]
    checks.append(("symmetric families", exact_sign_flip(mixed)["p_value"], 1.0))

    fourteen = [(f"f{i}", True, False) for i in range(14) for _ in range(2)]
    result = exact_sign_flip(fourteen)
    checks.append(("fourteen families", result["p_value"], 2 / 2 ** 14))
    checks.append(("fourteen family count", result["n_families"], 14))
    checks.append(("fourteen notice count", result["n_notices"], 28))

    failures = 0
    for name, actual, expected in checks:
        ok = actual == expected
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {actual} (expected {expected})")
    print("self-test passed" if failures == 0 else f"self-test failed: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(_self_test())
    print("usage: python scripts/sign_flip.py --self-test")
    sys.exit(2)
