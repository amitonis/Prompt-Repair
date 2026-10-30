"""Paired cluster bootstrap with the family as the resampling unit (Section 8).

Test notices are nested in 2-4 notice families that share template and
vocabulary, so notices are not independent. This module resamples families,
not notices, and carries every notice of a drawn family.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20261014
DEFAULT_CONFIDENCE = 0.95


def _percentile(ordered: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already sorted sequence."""
    if not ordered:
        raise ValueError("Cannot take a percentile of an empty sample")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_cluster_bootstrap(
    records: Iterable[tuple[str, bool, bool]],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, float | int]:
    """Bootstrap the paired strict-accuracy difference (arm A - arm B).

    ``records`` holds one ``(family_id, correct_a, correct_b)`` triple per
    notice, for the same notices under both arms. Families are drawn with
    replacement and all of a drawn family's notices are carried together.
    """
    grouped: dict[str, list[tuple[bool, bool]]] = {}
    for family_id, correct_a, correct_b in records:
        if not isinstance(family_id, str) or not family_id:
            raise ValueError("family_id must be a nonempty string")
        grouped.setdefault(family_id, []).append((bool(correct_a), bool(correct_b)))
    if not grouped:
        raise ValueError("At least one record is required")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")

    families = list(grouped.values())
    total = sum(len(rows) for rows in families)
    observed = sum(a - b for rows in families for a, b in rows) / total

    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_resamples):
        numerator = 0
        count = 0
        for _ in range(len(families)):
            rows = families[rng.randrange(len(families))]
            numerator += sum(a - b for a, b in rows)
            count += len(rows)
        deltas.append(numerator / count)
    deltas.sort()

    alpha = 1.0 - confidence
    low = _percentile(deltas, alpha / 2.0)
    high = _percentile(deltas, 1.0 - alpha / 2.0)
    share_at_or_below_zero = sum(d <= 0.0 for d in deltas) / n_resamples
    share_at_or_above_zero = sum(d >= 0.0 for d in deltas) / n_resamples
    p_value = min(1.0, 2.0 * min(share_at_or_below_zero, share_at_or_above_zero))

    return {
        "observed_delta": observed,
        "ci_low": low,
        "ci_high": high,
        "p_value": p_value,
        "n_notices": total,
        "n_families": len(families),
        "n_resamples": n_resamples,
        "seed": seed,
        "confidence": confidence,
    }
