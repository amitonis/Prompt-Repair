"""Resumable experiment runner: screening, sweeps, checkpointing (Sections 9, 14)."""

from __future__ import annotations

from typing import Any


def cache_key(
    model_id: str,
    model_revision: str | None,
    serving_library_version: str | None,
    quantization: str | None,
    rendered_prompt: str,
    notice: str,
    seed: int | None,
) -> str:
    """Cache key must change when any of these inputs changes (Section 14.6)."""
    import hashlib
    import json

    payload = json.dumps(
        [model_id, model_revision, serving_library_version, quantization, rendered_prompt, notice, seed],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def screen_defects(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Keep a defect only if it lowers development accuracy by the pre-registered margin."""
    raise NotImplementedError("Needs development runs from the adapter (Day 5)")


def run_sweep(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run one setting end to end with checkpointing and resume."""
    raise NotImplementedError("Wired to the adapter and storage (Day 7-9)")
