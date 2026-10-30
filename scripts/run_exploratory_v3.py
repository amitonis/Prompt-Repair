"""Compare the frozen baseline and v2 candidate on owner-reviewed draft 03."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "exploratory_draft_03"
RUNS = ROOT / "runs"
OUTPUT = RUNS / "exploratory_v3_comparison.json"
BASELINE_PROMPT = ROOT / "prompts" / "oracle.json"
CANDIDATE_PROMPT = ROOT / "prompts" / "exploratory_v2_boundary_conflict.json"
SEED = 20261030


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "attempted": 0}
    )
    for row in rows:
        bucket = by_category[row["category"]]
        bucket["attempted"] += 1
        bucket["correct"] += int(bool(row["score"]["record_correct"]))
    return {
        "correct": sum(bool(row["score"]["record_correct"]) for row in rows),
        "attempted": len(rows),
        "json_valid": sum(bool(row["score"]["json_valid"]) for row in rows),
        "schema_valid": sum(bool(row["score"]["schema_valid"]) for row in rows),
        "by_category": dict(by_category),
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}; preserve each run separately.")

    review = json.loads((DATA / "human_review.json").read_text(encoding="utf-8"))
    if review.get("run_authorized") is not True:
        raise RuntimeError("Owner authorization is missing from the v3 review record.")
    inputs = _jsonl(DATA / "inputs.jsonl")
    labels_path = DATA / "labels_reviewed.jsonl"
    if len(inputs) != 12 or not labels_path.is_file():
        raise ValueError("Expected 12 reviewed inputs and a reviewed-label file.")
    if any(row.get("split") != "exploratory_v3" for row in inputs):
        raise ValueError("Inputs do not carry the frozen exploratory_v3 split tag.")
    baseline = json.loads(BASELINE_PROMPT.read_text(encoding="utf-8"))["sections"]
    candidate = json.loads(CANDIDATE_PROMPT.read_text(encoding="utf-8"))["sections"]
    from src.prompt_renderer import render_prompt

    baseline_prompts = [render_prompt(baseline, row["notice"]) for row in inputs]
    candidate_prompts = [render_prompt(candidate, row["notice"]) for row in inputs]
    config = json.loads((ROOT / "config" / "experiment.json").read_text(encoding="utf-8"))
    model = config["models"]["extractor"]
    environment = config["environment"]

    import torch
    import transformers
    from transformers.utils import import_utils

    import_utils.is_accelerate_available.cache_clear()
    if not import_utils.is_accelerate_available():
        raise RuntimeError("Transformers cannot see accelerate in this runtime.")
    if not torch.cuda.is_available():
        raise RuntimeError("This exploratory comparison requires the Molab CUDA GPU.")
    gc.collect()
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 32 * 1024**3:
        raise RuntimeError(
            f"Not enough free GPU memory: {free_bytes / 1024**3:.1f} GiB free of "
            f"{total_bytes / 1024**3:.1f} GiB. Restart Molab before loading the model."
        )

    from src.inference import LocalTransformersAdapter

    adapter = LocalTransformersAdapter(
        model_id=model["hf_model_id"],
        revision=model["revision"],
        dtype=model["dtype"],
        quantization=model["quantization"],
        batch_size=min(4, int(environment["batch_size"])),
        max_new_tokens=256,
        device="cuda",
    )
    started = time.perf_counter()
    baseline_generations = adapter.generate_batch(
        baseline_prompts, run_id="exploratory-v3-oracle-baseline", seed=SEED,
        max_new_tokens=256,
    )
    candidate_generations = adapter.generate_batch(
        candidate_prompts, run_id="exploratory-v3-frozen-v2-candidate", seed=SEED,
        max_new_tokens=256,
    )
    elapsed_s = time.perf_counter() - started

    # Do not load reviewed gold labels until both arms have generated all outputs.
    labels = _jsonl(labels_path)
    if len(labels) != 12:
        raise ValueError("Expected exactly 12 reviewed labels.")
    labels_by_id = {row["id"]: row for row in labels}
    if {row["id"] for row in inputs} != set(labels_by_id):
        raise ValueError("Input and reviewed-label IDs do not match.")
    from src.evaluator import score_response

    arms: dict[str, list[dict[str, Any]]] = {
        "oracle_baseline": [],
        "frozen_v2_candidate": [],
    }
    for arm, generations in (
        ("oracle_baseline", baseline_generations),
        ("frozen_v2_candidate", candidate_generations),
    ):
        for item, generation in zip(inputs, generations):
            arms[arm].append({
                **item,
                "raw": generation.raw,
                "status": generation.status,
                "duration_s": generation.duration_s,
                "score": score_response(generation.raw, labels_by_id[item["id"]]["gold"]),
            })

    payload = {
        "study_scope": "post_hoc_exploratory_only",
        "warning": (
            "12 owner-reviewed synthetic cases; report raw counts only. Separate from the "
            "frozen primary study; this run does not support general claims."
        ),
        "notices": len(inputs),
        "seed": SEED,
        "elapsed_s": elapsed_s,
        "model": adapter.environment_record(),
        "artifacts_sha256": {
            "inputs": _sha256(DATA / "inputs.jsonl"),
            "labels": _sha256(DATA / "labels_reviewed.jsonl"),
            "baseline_prompt": _sha256(BASELINE_PROMPT),
            "candidate_prompt": _sha256(CANDIDATE_PROMPT),
        },
        "arm_summaries": {name: _summary(rows) for name, rows in arms.items()},
        "results": arms,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved exploratory output to {OUTPUT}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
