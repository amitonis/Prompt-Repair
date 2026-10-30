"""End-to-end D6 repair, selection, final evaluation, and reporting helpers.

This module is deliberately orchestration only. Prompt editing rules remain in
``optimizer`` and response scoring remains in ``evaluator``. The test split is
accepted by a separate function so it cannot accidentally influence editing or
selection.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .diagnostics import build_feedback_packet
from .evaluator import score_response
from .experiment_runner import run_generations
from .localizers import behav_localizer_rank, behav_neutralize, rule_localizer
from .optimizer import (
    parse_editor_json,
    run_rewrite_branch,
    run_span_edit_branch,
    run_targeted_branch,
)
from .prompt_renderer import render_prompt

SECTIONS = ("instruction", "context", "constraints", "output_format")


def _case_rows(
    adapter: Any,
    cases: Sequence[Mapping[str, Any]],
    sections: Mapping[str, str],
    *,
    setting_id: str,
    arm: str,
    run_id: str,
    cache_path: str | Path,
    seed: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    jobs = [
        {
            "setting_id": setting_id,
            "arm": arm,
            "notice_id": case["id"],
            "prompt": render_prompt(dict(sections), case["notice"]),
            "notice": case["notice"],
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
    by_id = {case["id"]: case for case in cases}
    rows: list[dict[str, Any]] = []
    for generation in generations:
        case = by_id[generation["notice_id"]]
        rows.append(
            {
                "id": case["id"],
                "split": case.get("split"),
                "category": case.get("category"),
                "notice": case["notice"],
                "gold": case["gold"],
                "raw": generation["raw"],
                "status": generation["status"],
                "cached": generation.get("cached", False),
                "duration_s": generation.get("duration_s"),
                "score": score_response(generation["raw"], case["gold"]),
            }
        )
    return rows


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores = [row["score"] for row in rows]
    return {
        "attempted": len(scores),
        "selection_records_correct": sum(bool(s["record_correct"]) for s in scores),
        "selection_invalid": sum(not bool(s["schema_valid"]) for s in scores),
        "json_valid": sum(bool(s["json_valid"]) for s in scores),
        "schema_valid": sum(bool(s["schema_valid"]) for s in scores),
        "field_correct": {
            field: sum(bool(s["field_correct"][field]) for s in scores)
            for field in ("event", "date", "venue")
        },
    }


def _editor_ask(adapter: Any, *, run_id: str, seed: int, max_new_tokens: int) -> Callable[[str], str]:
    def ask(prompt: str) -> str:
        record = adapter.generate(prompt, run_id=run_id, seed=seed, max_new_tokens=max_new_tokens)
        if record.status != "ok":
            raise RuntimeError(f"editor generation failed: {record.usage}")
        return record.raw

    return ask


def _llm_rank(ask: Callable[[str], str], packet: Mapping[str, Any]) -> list[str]:
    prompt = """You are diagnosing a reusable extraction prompt.
Rank these four prompt sections from most likely to least likely to cause the
observed failures. Use only the error codes and examples. Do not propose an
answer to any notice. Return exactly one JSON object with key `ranking`, whose
value is a list containing each section exactly once.

SECTIONS: ["instruction", "context", "constraints", "output_format"]
FEEDBACK:
""" + json.dumps(packet, indent=2, sort_keys=True)
    proposal = parse_editor_json(ask(prompt))
    ranking = proposal.get("ranking")
    if not isinstance(ranking, list) or set(ranking) != set(SECTIONS) or len(ranking) != len(SECTIONS):
        raise ValueError("LLM localizer must return a permutation of the four sections")
    return [str(section) for section in ranking]


def run_d6_repair_sweep(
    adapter: Any,
    *,
    task_contract: str,
    defective_sections: Mapping[str, str],
    feedback_cases: Sequence[Mapping[str, Any]],
    selection_cases: Sequence[Mapping[str, Any]],
    cache_path: str | Path,
    run_id: str = "d6-repair",
    setting_id: str = "D6",
    seed: int = 20261014,
    extractor_max_new_tokens: int = 256,
    editor_max_new_tokens: int = 1024,
    max_attempts: int = 2,
    include_llm_localizer: bool = True,
) -> dict[str, Any]:
    """Run D6 feedback, editing branches, and selection-set acceptance.

    The returned object is JSON serializable and contains all raw extraction
    outputs, feedback, editor attempts, candidate prompts, and selection
    metrics. No test notice is accepted by this function.
    """
    started = time.perf_counter()
    if set(defective_sections) != set(SECTIONS):
        raise ValueError("defective_sections must contain exactly the four prompt sections")
    feedback_rows = _case_rows(
        adapter, feedback_cases, defective_sections,
        setting_id=setting_id, arm="defective", run_id=f"{run_id}-feedback",
        cache_path=cache_path, seed=seed, max_new_tokens=extractor_max_new_tokens,
    )
    selection_rows = _case_rows(
        adapter, selection_cases, defective_sections,
        setting_id=setting_id, arm="defective", run_id=f"{run_id}-selection",
        cache_path=cache_path, seed=seed, max_new_tokens=extractor_max_new_tokens,
    )
    current_metrics = _metrics(selection_rows)
    packet = build_feedback_packet(feedback_rows, feedback_subset_size=len(feedback_rows))
    ask = _editor_ask(adapter, run_id=f"{run_id}-editor", seed=seed, max_new_tokens=editor_max_new_tokens)
    score = lambda sections: _metrics(
        _case_rows(
            adapter, selection_cases, sections,
            setting_id=setting_id, arm="candidate", run_id=f"{run_id}-candidate",
            cache_path=cache_path, seed=seed, max_new_tokens=extractor_max_new_tokens,
        )
    )

    rewrite = run_rewrite_branch(
        dict(defective_sections), current_metrics, ask=ask, score=score,
        task_contract=task_contract, feedback_packet=packet, max_attempts=max_attempts,
    )
    span = run_span_edit_branch(
        dict(defective_sections), current_metrics, ask=ask, score=score,
        task_contract=task_contract, feedback_packet=packet, max_attempts=max_attempts,
    )
    targeted = run_targeted_branch(
        dict(defective_sections), current_metrics, ask=ask, score=score,
        task_contract=task_contract, feedback_packet=packet, max_attempts=max_attempts,
    )

    rankings = {"L-Rule": rule_localizer(packet)}
    if include_llm_localizer:
        try:
            rankings["L-LLM"] = _llm_rank(ask, packet)
        except (RuntimeError, ValueError) as exc:
            rankings["L-LLM"] = {"error": str(exc)}

    failing_feedback = [row for row in feedback_rows if not row["score"]["record_correct"]]
    output_changes: dict[str, list[bool]] = {}
    for section in SECTIONS:
        neutralized = behav_neutralize(dict(defective_sections), section)
        neutral_rows = _case_rows(
            adapter, [
                {"id": row["id"], "notice": row["notice"], "gold": row["gold"]}
                for row in failing_feedback
            ], neutralized,
            setting_id=setting_id, arm=f"L-Behav-{section}", run_id=f"{run_id}-behav",
            cache_path=cache_path, seed=seed, max_new_tokens=extractor_max_new_tokens,
        )
        original_by_id = {row["id"]: row["raw"] for row in failing_feedback}
        output_changes[section] = [row["raw"] != original_by_id[row["id"]] for row in neutral_rows]
    rankings["L-Behav"] = behav_localizer_rank(output_changes)

    patches = targeted.patches
    arms: dict[str, dict[str, Any]] = {
        "defective": {"sections": dict(defective_sections), "metrics": current_metrics, "accepted": False},
        "rewrite": {"sections": rewrite.sections, "metrics": rewrite.metrics, "accepted": rewrite.accepted},
        "span": {"sections": span.sections, "metrics": span.metrics, "accepted": span.accepted},
    }
    for localizer_name, ranking in rankings.items():
        if isinstance(ranking, dict):
            continue
        section = ranking[0]
        patch = patches.get(section, {})
        arms[localizer_name] = {
            "sections": patch.get("sections", dict(defective_sections)),
            "metrics": patch.get("metrics", current_metrics),
            "accepted": bool(patch.get("accepted", False)),
            "selected_section": section,
        }
    for name, section in (("oracle-section", "output_format"), ("random-section-mean", None)):
        if section is not None:
            patch = patches.get(section, {})
            arms[name] = {
                "sections": patch.get("sections", dict(defective_sections)),
                "metrics": patch.get("metrics", current_metrics),
                "accepted": bool(patch.get("accepted", False)),
                "selected_section": section,
            }
        else:
            arms[name] = {"section_metrics": {s: patches.get(s, {}).get("metrics", current_metrics) for s in SECTIONS}}

    return {
        "arm": "d6_repair_sweep",
        "setting_id": setting_id,
        "defect_id": "D6",
        "elapsed_s": time.perf_counter() - started,
        "feedback": {"records": feedback_rows, "metrics": _metrics(feedback_rows)},
        "selection": {"records": selection_rows, "metrics": current_metrics},
        "feedback_packet": packet,
        "localizer_rankings": rankings,
        "output_changes": output_changes,
        "branches": {
            "rewrite": rewrite.__dict__,
            "span": span.__dict__,
            "targeted": targeted.__dict__,
        },
        "arms": arms,
    }


def evaluate_locked_arms(
    adapter: Any,
    locked_arms: Mapping[str, Mapping[str, Any]],
    test_cases: Sequence[Mapping[str, Any]],
    *,
    cache_path: str | Path,
    run_id: str = "final-test",
    setting_id: str = "D6",
    seed: int = 20261014,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    """Evaluate already locked prompts on the test split exactly once."""
    started = time.perf_counter()
    results: dict[str, Any] = {}
    for arm_name, arm in locked_arms.items():
        sections = arm.get("sections")
        if not isinstance(sections, Mapping) or set(sections) != set(SECTIONS):
            raise ValueError(f"Locked arm {arm_name} does not contain four prompt sections")
        rows = _case_rows(
            adapter, test_cases, sections,
            setting_id=setting_id, arm=arm_name, run_id=f"{run_id}-{arm_name}",
            cache_path=cache_path, seed=seed, max_new_tokens=max_new_tokens,
        )
        metrics = _metrics(rows)
        results[arm_name] = {"metrics": metrics, "records": rows, "sections": dict(sections)}
    return {"arm": "final_test", "setting_id": setting_id, "elapsed_s": time.perf_counter() - started, "results": results}
