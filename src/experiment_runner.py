"""Resumable experiment runner: screening, sweeps, checkpointing (Sections 9, 14)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

MANIFEST_FIELDS = (
    "git_commit",
    "model_id",
    "model_revision",
    "serving_library",
    "serving_library_version",
    "torch_version",
    "dtype",
    "quantization",
    "gpu_name",
    "cuda_version",
    "driver_version",
    "batch_size",
    "seed",
    "started_at_utc",
)

# Session-local timestamps differ by design and are not deviations.
VOLATILE_MANIFEST_FIELDS = frozenset({"started_at_utc"})


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

    payload = json.dumps(
        [model_id, model_revision, serving_library_version, quantization, rendered_prompt, notice, seed],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def session_manifest(
    *,
    git_commit: str,
    model_id: str,
    model_revision: str | None = None,
    serving_library: str | None = None,
    serving_library_version: str | None = None,
    torch_version: str | None = None,
    dtype: str | None = None,
    quantization: str | None = None,
    gpu_name: str | None = None,
    cuda_version: str | None = None,
    driver_version: str | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    started_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the per-session record required by analysis_plan.md Section 9."""
    if not isinstance(git_commit, str) or not git_commit.strip():
        raise ValueError("git_commit must be a nonempty string")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonempty string")
    if batch_size is not None and (not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError("batch_size must be a positive integer or None")

    manifest = {
        "git_commit": git_commit,
        "model_id": model_id,
        "model_revision": model_revision,
        "serving_library": serving_library,
        "serving_library_version": serving_library_version,
        "torch_version": torch_version,
        "dtype": dtype,
        "quantization": quantization,
        "gpu_name": gpu_name,
        "cuda_version": cuda_version,
        "driver_version": driver_version,
        "batch_size": batch_size,
        "seed": seed,
        "started_at_utc": started_at_utc or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    assert tuple(manifest) == MANIFEST_FIELDS
    return manifest


def manifest_diff(previous: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the fields that changed between sessions, for the deviation log."""
    changes: list[dict[str, Any]] = []
    for field in MANIFEST_FIELDS:
        if field in VOLATILE_MANIFEST_FIELDS:
            continue
        before = previous.get(field)
        after = current.get(field)
        if before != after:
            changes.append({"field": field, "previous": before, "current": after})
    return changes


def write_session_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Write one session manifest as JSON, creating parent directories."""
    if not isinstance(manifest, dict) or set(manifest) != set(MANIFEST_FIELDS):
        raise ValueError(f"manifest must contain exactly {list(MANIFEST_FIELDS)}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    """Index cached generations by cache key; the first record is canonical."""
    index: dict[str, dict[str, Any]] = {}
    source = Path(path)
    if not source.exists():
        return index
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            key = record["cache_key"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{source}:{number}: invalid cache record ({exc})") from exc
        if not isinstance(key, str) or not key:
            raise ValueError(f"{source}:{number}: cache_key must be a nonempty string")
        index.setdefault(key, record)
    return index


def append_cache(path: str | Path, record: dict[str, Any]) -> bool:
    """Append a generation record unless its key is cached. Returns True if written."""
    if not isinstance(record, dict):
        raise TypeError("record must be a dict")
    key = record.get("cache_key")
    if not isinstance(key, str) or not key:
        raise ValueError("record must contain a nonempty cache_key")

    destination = Path(path)
    if key in load_cache(destination):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return True


def missing_jobs(
    cache_path: str | Path,
    jobs: Iterable[tuple[str, Any]],
) -> list[tuple[str, Any]]:
    """Return the jobs whose cache key is absent, preserving input order."""
    cached = load_cache(cache_path)
    pending: list[tuple[str, Any]] = []
    for job in jobs:
        if not isinstance(job, Sequence) or len(job) != 2:
            raise ValueError("each job must be a (cache_key, payload) pair")
        key = job[0]
        if not isinstance(key, str) or not key:
            raise ValueError("job cache_key must be a nonempty string")
        if key not in cached:
            pending.append(job)
    return pending


@dataclass(frozen=True)
class ScreenResult:
    """One defect's screening outcome on the development notices (Section 3)."""

    defect_id: str
    kept: bool
    drop_points: float
    flips: tuple[str, ...]
    target_size: int
    reason: str


JOB_FIELDS = ("setting_id", "arm", "notice_id", "prompt", "notice")


def run_generations(
    adapter: Any,
    jobs: Sequence[Mapping[str, Any]],
    cache_path: str | Path,
    *,
    run_id: str,
    seed: int | None = None,
    max_new_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Run pending generations in job order, reusing the append-only cache.

    A job needs setting_id, arm, notice_id, prompt and notice. Only successful
    generations are cached; failures are returned with status "error" and are
    retried on the next run.
    """
    cached = load_cache(cache_path)
    results: dict[str, dict[str, Any]] = {}
    pending: list[tuple[str, Mapping[str, Any]]] = []
    pending_keys: set[str] = set()
    order: list[str] = []

    for job in jobs:
        missing = [field for field in JOB_FIELDS if not job.get(field)]
        if missing:
            raise ValueError(f"job is missing fields: {missing}")
        key = cache_key(
            adapter.model_id,
            adapter.revision,
            getattr(adapter, "serving_library_version", None),
            adapter.quantization,
            job["prompt"],
            job["notice"],
            seed,
        )
        order.append(key)
        if key in cached:
            results.setdefault(key, {**cached[key], "cached": True})
        elif key not in pending_keys:
            pending.append((key, job))
            pending_keys.add(key)

    if pending:
        settings: dict[str, Any] = {"run_id": run_id, "seed": seed}
        if max_new_tokens is not None:
            settings["max_new_tokens"] = max_new_tokens
        generated = adapter.generate_batch([job["prompt"] for _, job in pending], **settings)
        if len(generated) != len(pending):
            raise RuntimeError(f"adapter returned {len(generated)} records for {len(pending)} jobs")
        for (key, job), record in zip(pending, generated):
            payload = {
                "cache_key": key,
                "run_id": run_id,
                "setting_id": job["setting_id"],
                "arm": job["arm"],
                "notice_id": job["notice_id"],
                "model_id": record.model_id,
                "model_revision": record.model_revision,
                "prompt_sha256": record.prompt_hash,
                "raw": record.raw,
                "status": record.status,
                "duration_s": record.duration_s,
                "usage": record.usage,
                "serving_library": record.serving_library,
                "quantization": record.quantization,
                "seed": record.seed,
                "replicate": record.replicate,
            }
            if record.status == "ok":
                append_cache(cache_path, payload)
            results[key] = {**payload, "cached": False}

    return [results[key] for key in order]


def screen_defects(
    oracle_correct: Mapping[str, bool],
    defect_correct: Mapping[str, Mapping[str, bool]],
    *,
    target_notices: Mapping[str, Sequence[str]],
    min_drop_points: float = 15.0,
    thin_min_drop_points: float = 10.0,
    thin_max_target: int = 2,
    thin_defect_ids: Iterable[str] = ("D3",),
    min_flips_by_defect: Mapping[str, int] = MappingProxyType({"D2": 3}),
) -> list[ScreenResult]:
    """Apply the pre-registered defect-retention rules on development notices.

    Standard rule: keep if strict accuracy drops by at least 15 points versus
    the oracle. Thin-category rule (D3 only): with fewer than 3 targeted
    notices, keep if the drop is at least 10 points and every targeted notice
    flips. D2: keep only if at least 3 of its 4 targeted notices flip.
    """
    if not oracle_correct:
        raise ValueError("oracle_correct must not be empty")
    oracle_ids = set(oracle_correct)
    thin_ids = set(thin_defect_ids)

    results: list[ScreenResult] = []
    for defect_id in sorted(defect_correct):
        scores = defect_correct[defect_id]
        if set(scores) != oracle_ids:
            raise ValueError(f"defect {defect_id} must score every development notice")
        targets = list(target_notices.get(defect_id, ()))
        if not targets:
            raise ValueError(f"defect {defect_id} needs at least one targeted notice")
        if not set(targets) <= oracle_ids:
            raise ValueError(f"defect {defect_id} names unknown targeted notices")

        oracle_hits = sum(1 for value in oracle_correct.values() if value)
        defect_hits = sum(1 for value in scores.values() if value)
        drop_points = 100 * (oracle_hits - defect_hits) / len(oracle_ids)
        flips = tuple(n for n in targets if oracle_correct[n] and not scores[n])

        if defect_id in thin_ids and len(targets) <= thin_max_target:
            kept = drop_points >= thin_min_drop_points and len(flips) == len(targets)
            reason = "thin-category rule" if kept else "thin-category rule not met"
        elif defect_id in min_flips_by_defect:
            needed = min_flips_by_defect[defect_id]
            kept = len(flips) >= needed
            reason = f"flip rule ({len(flips)}/{len(targets)}, needs {needed})"
        else:
            kept = drop_points >= min_drop_points
            reason = "standard 15-point rule" if kept else "drop below 15 points"

        results.append(
            ScreenResult(
                defect_id=defect_id,
                kept=kept,
                drop_points=drop_points,
                flips=flips,
                target_size=len(targets),
                reason=reason,
            )
        )
    return results


def has_feedback_signal(defect_correct: Mapping[str, bool], feedback_ids: Sequence[str]) -> bool:
    """True if the defective prompt fails at least one feedback notice."""
    if not feedback_ids:
        raise ValueError("feedback_ids must not be empty")
    missing = [n for n in feedback_ids if n not in defect_correct]
    if missing:
        raise ValueError(f"feedback notices missing from scores: {missing}")
    return any(not defect_correct[n] for n in feedback_ids)


def oracle_ceiling_met(oracle_correct: Mapping[str, bool], minimum: float = 0.85) -> bool:
    """Model is included only if the oracle reaches the development ceiling."""
    if not oracle_correct:
        raise ValueError("oracle_correct must not be empty")
    return sum(1 for value in oracle_correct.values() if value) / len(oracle_correct) >= minimum


def original_prompt_has_headroom(original_correct: Mapping[str, bool], minimum_failures: int = 3) -> bool:
    """The original prompt is included only with enough development failures."""
    if not original_correct:
        raise ValueError("original_correct must not be empty")
    failures = sum(1 for value in original_correct.values() if not value)
    return failures >= minimum_failures


def run_sweep(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility entry point for the implemented D6 repair sweep.

    The orchestration lives in ``src.pipeline`` to keep cache primitives and
    the experiment protocol separate. This entry point preserves the original
    public name used by notebooks and future callers.
    """
    from .pipeline import run_d6_repair_sweep

    return run_d6_repair_sweep(*args, **kwargs)
