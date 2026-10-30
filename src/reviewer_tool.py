"""Tier A reviewer prompt repair on the fixed college-notice development data.

This entry point deliberately loads only feedback and selection splits. It never
loads or evaluates the frozen test split.
"""

from __future__ import annotations

import difflib
import gc
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .defects import SECTIONS, load_prompt
from .diagnostics import build_feedback_packet
from .evaluator import score_response
from .experiment_runner import run_generations
from .localizers import rule_localizer
from .optimizer import run_rewrite_branch, run_targeted_branch
from .prompt_renderer import render_prompt
from .storage import load_subset

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "experiment.json"
CONTRACT_PATH = ROOT / "task_contract.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reviewer_prompt(path: str | Path) -> dict[str, str]:
    """Load sectioned JSON or plain text; omitted sections use task defaults.

    JSON may be either {"sections": {...}} or a section mapping. Plain text is
    treated as the instruction section. Missing sections are filled from the
    canonical notice-task prompt so every repair is scored against the same
    task contract and output schema.
    """
    prompt_path = Path(path)
    raw = prompt_path.read_text(encoding="utf-8")
    defaults = load_prompt("oracle")
    if prompt_path.suffix.lower() == ".json":
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Prompt JSON must contain an object.")
        proposed = data.get("sections", data)
        if not isinstance(proposed, dict):
            raise ValueError("Prompt JSON 'sections' must be an object.")
    else:
        proposed = {"instruction": raw.strip()}

    unknown = sorted(set(proposed) - set(SECTIONS))
    if unknown:
        raise ValueError(f"Unknown prompt sections: {unknown}")
    sections = dict(defaults)
    for name, text in proposed.items():
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Prompt section {name!r} must be a nonempty string.")
        sections[name] = text.strip()
    if not any(name in proposed for name in SECTIONS):
        raise ValueError("Prompt must provide at least one recognized section.")
    return sections


def _evaluate(
    adapter: Any,
    cases: Sequence[Mapping[str, Any]],
    sections: Mapping[str, str],
    *,
    split: str,
    arm: str,
    run_id: str,
    cache_path: Path,
    seed: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    # This function is called only with load_subset("feedback") or
    # load_subset("selection"); test cases are rejected defensively.
    if split not in {"feedback", "selection"}:
        raise ValueError("Reviewer repair may use development splits only.")
    jobs = [
        {
            "setting_id": "reviewer_prompt_tier_a",
            "arm": arm,
            "notice_id": str(case["id"]),
            "prompt": render_prompt(dict(sections), str(case["notice"])),
            "notice": str(case["notice"]),
        }
        for case in cases
    ]
    generations = run_generations(
        adapter,
        jobs,
        cache_path,
        run_id=run_id,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )
    by_id = {str(case["id"]): case for case in cases}
    rows = []
    for generation in generations:
        case = by_id[generation["notice_id"]]
        rows.append(
            {
                "id": case["id"],
                "family_id": case.get("family_id"),
                "split": split,
                "category": case.get("category"),
                "notice": case["notice"],
                "gold": case["gold"],
                "raw": generation["raw"],
                "status": generation["status"],
                "cached": generation.get("cached", False),
                "score": score_response(generation["raw"], case["gold"]),
            }
        )
    return rows


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores = [row["score"] for row in rows]
    attempted = len(scores)
    correct = sum(bool(score["record_correct"]) for score in scores)
    return {
        "attempted": attempted,
        "record_correct": correct,
        "accuracy": correct / attempted if attempted else None,
        "json_valid": sum(bool(score["json_valid"]) for score in scores),
        "schema_valid": sum(bool(score["schema_valid"]) for score in scores),
        "field_correct": {
            field: sum(bool(score["field_correct"][field]) for score in scores)
            for field in ("event", "date", "venue")
        },
        "issue_counts": dict(
            Counter(issue for score in scores for issue in score.get("issues", []))
        ),
    }


def _editor_ask(adapter: Any, run_id: str, seed: int, max_new_tokens: int):
    def ask(prompt: str) -> str:
        record = adapter.generate(
            prompt,
            run_id=f"{run_id}-editor",
            seed=seed,
            max_new_tokens=max_new_tokens,
        )
        if record.status != "ok":
            raise RuntimeError(f"Prompt editor failed: {record.usage}")
        return record.raw

    return ask


def _selection_metrics(
    adapter: Any,
    cases: Sequence[Mapping[str, Any]],
    sections: dict[str, str],
    *,
    arm: str,
    run_id: str,
    cache_path: Path,
    seed: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    rows = _evaluate(
        adapter,
        cases,
        sections,
        split="selection",
        arm=arm,
        run_id=run_id,
        cache_path=cache_path,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )
    metrics = _metrics(rows)
    metrics["selection_records_correct"] = metrics["record_correct"]
    metrics["selection_invalid"] = metrics["attempted"] - metrics["schema_valid"]
    return metrics


def _candidate_details(result: Any, original: Mapping[str, str]) -> dict[str, Any]:
    sections = dict(result.sections)
    changed = [section for section in SECTIONS if sections[section] != original[section]]
    if hasattr(result, "patches"):
        attempts = result.attempts
        candidates = []
        for section, patch in result.patches.items():
            accepted = bool(patch.get("accepted", False))
            reason = patch.get("reason")
            if not accepted and reason is None and patch.get("metrics"):
                reason = "selection_acceptance_rule_not_met"
            candidates.append(
                {
                    "section": section,
                    "accepted": accepted,
                    "reason": reason,
                    "metrics": patch.get("metrics"),
                }
            )
        accepted_section = result.accepted_section
        stopped_reason = result.stopped_reason
    else:
        attempts = result.attempts
        candidates = [
            {
                "attempt": attempt.get("attempt"),
                "accepted": bool(attempt.get("accepted", False)),
                "reason": attempt.get("reason"),
                "metrics": attempt.get("metrics"),
            }
            for attempt in attempts
        ]
        accepted_section = changed[0] if len(changed) == 1 else None
        stopped_reason = result.stopped_reason
    return {
        "accepted": bool(result.accepted),
        "accepted_section": accepted_section,
        "changed_sections": changed,
        "stopped_reason": stopped_reason,
        "selection_metrics": result.metrics,
        "candidates": candidates,
        "attempts": attempts,
        "sections": sections,
    }


def run_repair(
    prompt_file: str | Path,
    *,
    mode: str = "both",
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    device: str = "cuda",
    seed: int = 20261030,
    max_attempts: int = 2,
) -> Path:
    """Repair a notice-task prompt using feedback and selection only."""
    if mode not in {"targeted", "rewrite", "both"}:
        raise ValueError("mode must be targeted, rewrite, or both")
    if max_attempts < 1 or max_attempts > 2:
        raise ValueError("max_attempts must be 1 or 2")
    prompt_path = Path(prompt_file).resolve()
    if not prompt_path.is_file():
        raise FileNotFoundError(prompt_path)
    base_id = run_id or f"repair-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", base_id):
        raise ValueError("run_id may contain only letters, numbers, dot, underscore, and hyphen")
    destination = Path(output_dir) if output_dir else ROOT / "runs" / "reviewer_repairs" / base_id
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=False)

    sections = load_reviewer_prompt(prompt_path)
    feedback_cases = load_subset("feedback")
    selection_cases = load_subset("selection")
    if not feedback_cases or not selection_cases:
        raise ValueError("Feedback and selection development sets must both be nonempty.")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model_config = config["models"]["extractor"]
    environment = config["environment"]
    extractor_tokens = int(model_config["max_new_tokens"])
    editor_tokens = int(config["models"]["editor"]["max_new_tokens"])
    cache_path = destination / "generation_cache.jsonl"

    import torch
    from transformers.utils import import_utils

    import_utils.is_accelerate_available.cache_clear()
    if device.startswith("cuda"):
        if not import_utils.is_accelerate_available():
            raise RuntimeError("Transformers cannot see accelerate in this runtime.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; attach a GPU runtime or pass --device cpu.")
        gc.collect()
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if free_bytes < 32 * 1024**3:
            raise RuntimeError(
                f"Only {free_bytes / 1024**3:.1f} GiB GPU memory is free of "
                f"{total_bytes / 1024**3:.1f} GiB; restart the runtime before loading the model."
            )

    from .inference import LocalTransformersAdapter

    adapter = LocalTransformersAdapter(
        model_id=model_config["hf_model_id"],
        revision=model_config["revision"],
        dtype=model_config["dtype"],
        quantization=model_config["quantization"],
        batch_size=min(8, int(environment["batch_size"])),
        max_new_tokens=extractor_tokens,
        device=device,
    )

    started = time.perf_counter()
    feedback_rows = _evaluate(
        adapter,
        feedback_cases,
        sections,
        split="feedback",
        arm="reviewer_prompt_before",
        run_id=f"{base_id}-feedback-before",
        cache_path=cache_path,
        seed=seed,
        max_new_tokens=extractor_tokens,
    )
    selection_rows = _evaluate(
        adapter,
        selection_cases,
        sections,
        split="selection",
        arm="reviewer_prompt_before",
        run_id=f"{base_id}-selection-before",
        cache_path=cache_path,
        seed=seed,
        max_new_tokens=extractor_tokens,
    )
    feedback_packet = build_feedback_packet(feedback_rows, len(feedback_rows))
    current_selection = _metrics(selection_rows)
    current_selection["selection_records_correct"] = current_selection["record_correct"]
    current_selection["selection_invalid"] = (
        current_selection["attempted"] - current_selection["schema_valid"]
    )
    score = lambda candidate: _selection_metrics(
        adapter,
        selection_cases,
        dict(candidate),
        arm="reviewer_candidate",
        run_id=f"{base_id}-candidate",
        cache_path=cache_path,
        seed=seed,
        max_new_tokens=extractor_tokens,
    )
    ask = _editor_ask(adapter, base_id, seed, editor_tokens)
    localizer_ranking = rule_localizer(feedback_packet)
    repair_results: dict[str, dict[str, Any]] = {}

    if mode in {"targeted", "both"}:
        targeted = run_targeted_branch(
            dict(sections),
            current_selection,
            ask=ask,
            score=score,
            task_contract=CONTRACT_PATH.read_text(encoding="utf-8"),
            feedback_packet=feedback_packet,
            target_sections=(localizer_ranking[0],),
            max_attempts=max_attempts,
        )
        repair_results["targeted"] = _candidate_details(targeted, sections)
    if mode in {"rewrite", "both"}:
        rewrite = run_rewrite_branch(
            dict(sections),
            current_selection,
            ask=ask,
            score=score,
            task_contract=CONTRACT_PATH.read_text(encoding="utf-8"),
            feedback_packet=feedback_packet,
            max_attempts=max_attempts,
        )
        repair_results["rewrite"] = _candidate_details(rewrite, sections)

    for name, result in repair_results.items():
        result["feedback_metrics_after"] = _metrics(
            _evaluate(
                adapter,
                feedback_cases,
                result["sections"],
                split="feedback",
                arm=f"reviewer_prompt_after_{name}",
                run_id=f"{base_id}-feedback-after-{name}",
                cache_path=cache_path,
                seed=seed,
                max_new_tokens=extractor_tokens,
            )
        )
        result["selection_metrics_after"] = _selection_metrics(
            adapter,
            selection_cases,
            result["sections"],
            arm=f"reviewer_prompt_after_{name}",
            run_id=f"{base_id}-selection-after-{name}",
            cache_path=cache_path,
            seed=seed,
            max_new_tokens=extractor_tokens,
        )

    elapsed = time.perf_counter() - started
    before = {"feedback": _metrics(feedback_rows), "selection": current_selection}
    report = {
        "scope": "tier_a_notice_task_development_only",
        "warning": "Uses feedback and selection development data only. Frozen test data are never loaded or scored by this command.",
        "run_id": base_id,
        "mode": mode,
        "prompt_file": str(prompt_path),
        "prompt_sha256": _sha256(prompt_path),
        "task_contract_sha256": _sha256(CONTRACT_PATH),
        "seed": seed,
        "elapsed_s": elapsed,
        "model": adapter.environment_record(),
        "localizer": {"method": "L-Rule", "ranking": localizer_ranking},
        "before": before,
        "repairs": repair_results,
        "feedback_packet": feedback_packet,
        "feedback_records_before": feedback_rows,
        "selection_records_before": selection_rows,
    }
    (destination / "prompt_before.json").write_text(
        json.dumps({"sections": sections}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for name, result in repair_results.items():
        (destination / f"prompt_after_{name}.json").write_text(
            json.dumps({"sections": result["sections"]}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        before_text = json.dumps(sections, indent=2, ensure_ascii=False).splitlines()
        after_text = json.dumps(result["sections"], indent=2, ensure_ascii=False).splitlines()
        diff = difflib.unified_diff(
            before_text,
            after_text,
            fromfile="prompt_before.json",
            tofile=f"prompt_after_{name}.json",
            lineterm="",
        )
        (destination / f"diff_{name}.patch").write_text("\n".join(diff) + "\n", encoding="utf-8")
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (destination / "report.md").write_text(_render_report(report), encoding="utf-8")
    return destination


def _render_report(report: Mapping[str, Any]) -> str:
    before = report["before"]
    lines = [
        "# Reviewer prompt repair report",
        "",
        "**Scope:** Tier A fixed notice task; feedback and selection development splits only. The frozen test split was not loaded.",
        "",
        f"- Run: `{report['run_id']}`",
        f"- Mode: `{report['mode']}`",
        f"- Prompt input: `{report['prompt_file']}`",
        f"- Elapsed: {report['elapsed_s']:.1f} seconds",
        f"- Localizer ranking: {', '.join(report['localizer']['ranking'])}",
        "",
        "## Before",
        "",
        "| Split | Exact records | JSON valid | Schema valid |",
        "| --- | ---: | ---: | ---: |",
        f"| Feedback | {before['feedback']['record_correct']}/{before['feedback']['attempted']} | {before['feedback']['json_valid']}/{before['feedback']['attempted']} | {before['feedback']['schema_valid']}/{before['feedback']['attempted']} |",
        f"| Selection | {before['selection']['record_correct']}/{before['selection']['attempted']} | {before['selection']['json_valid']}/{before['selection']['attempted']} | {before['selection']['schema_valid']}/{before['selection']['attempted']} |",
        "",
        "## Candidate results",
        "",
        "| Method | Accepted | Feedback before → after | Selection before → after | Section changed |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for name, result in report["repairs"].items():
        feedback_after = result["feedback_metrics_after"]
        selection_after = result["selection_metrics_after"]
        changed = ", ".join(result["changed_sections"]) or "none (original retained)"
        lines.append(
            f"| {name} | {'yes' if result['accepted'] else 'no'} | "
            f"{before['feedback']['record_correct']}/{before['feedback']['attempted']} → "
            f"{feedback_after['record_correct']}/{feedback_after['attempted']} | "
            f"{before['selection']['record_correct']}/{before['selection']['attempted']} → "
            f"{selection_after['record_correct']}/{selection_after['attempted']} | {changed} |"
        )
    lines += ["", "## Candidate decisions", ""]
    for name, result in report["repairs"].items():
        lines += [f"### {name}", ""]
        if result["candidates"]:
            lines += ["| Candidate | Accepted | Reason | Selection correct |", "| --- | --- | --- | ---: |"]
            for index, candidate in enumerate(result["candidates"], 1):
                metrics = candidate.get("metrics") or {}
                correct = metrics.get("selection_records_correct", "—")
                attempted = metrics.get("attempted", before["selection"]["attempted"])
                label = candidate.get("section") or candidate.get("attempt") or index
                reason = candidate.get("reason") or candidate.get("stopped_reason") or "scored by selection rule"
                lines.append(
                    f"| {label} | {'yes' if candidate.get('accepted') else 'no'} | {reason} | {correct}/{attempted} |"
                )
        else:
            lines.append("No valid edit candidate was produced.")
        lines.append("")
        lines.append(f"Diff: `diff_{name}.patch`; prompt file: `prompt_after_{name}.json`.")
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        "Acceptance uses the development selection split: a repair must strictly improve exact-record accuracy and must not increase the number of schema-invalid responses. This report is a development result, not a test-set claim. In `both` mode, targeted patch and full rewrite are shown side by side; their comparison is descriptive.",
        "",
    ]
    return "\n".join(lines)
