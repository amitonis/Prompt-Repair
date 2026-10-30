# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "accelerate",
#     "torch",
#     "transformers",
# ]
# ///
import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Day 1 — molab environment setup

        Purpose: record GPU details, download the pre-registered model,
        measure throughput, and verify Drive persistence.

        **Rules**

        - Never paste HF or GitHub tokens into this notebook: molab notebooks
          are public but undiscoverable.
        - Sessions stop after ~12 hours; keep today short.
        - `MODEL_ID` below must match `config/experiment.json`. The record cell
          writes `runs/day1/environment.json`; copy those values into
          `config/experiment.json` and the `TO_FILL` markers in
          `analysis_plan.md` Section 1, then re-run `bash scripts/hash_frozen.sh`.
        """
    )
    return


@app.cell
def _():
    import subprocess

    nvidia_smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(nvidia_smi.stdout)
    if nvidia_smi.stderr:
        print(nvidia_smi.stderr)
    return


@app.cell
def _(mo):
    mo.md(
        """
        **VRAM math:** bf16 is ~2 bytes per parameter (7B ≈ 16 GB, 14B ≈ 28 GB).
        Revision 2 pre-registers `Qwen/Qwen2.5-14B-Instruct`; do not switch to
        a different size without logging a deviation.
        """
    )
    return


@app.cell
def _():
    MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    print(torch.cuda.get_device_name(0))
    print(f"Loaded {MODEL_ID}")
    return MODEL_ID, model, tok, torch


@app.cell
def _(mo):
    mo.md(
        """
        ## Throughput benchmark

        ~200 short prompts, batched greedy generation. If it OOMs, halve
        `BATCH_SIZE`. Record **prompts/s**; sweep time ≈ total generations ÷
        prompts per second. Target: full sweep ≤ 6 hours; if larger, shrink
        per the methodology guide (fewer repeats → fewer defects → one model).
        The batch size that survives here becomes `B` in the manifest.
        """
    )
    return


@app.cell
def _(model, tok):
    import time

    BATCH_SIZE = 16
    MAX_NEW_TOKENS = 64
    N_PROMPTS = 200

    template = (
        "[INSTRUCTION]\n"
        "Extract the name, event date, and venue of the single target college "
        "event described in the supplied notice.\n\n"
        "[CONSTRAINTS]\n"
        "Use only information explicitly stated in the notice. Copy values "
        "verbatim. Return null for any field the notice does not state.\n\n"
        "[OUTPUT_FORMAT]\n"
        "Return exactly one JSON object and no other text, with exactly the "
        "keys event, date, and venue.\n\n"
        "[NOTICE]\n"
        "{notice}"
    )
    clubs = ["AI Club", "Robotics Club", "Coding Society", "Debate Union"]
    events = ["Prompt Workshop", "Design Seminar", "Data Symposium", "Career Fair"]
    dates = ["20 October 2026", "3 November 2026", "12 December 2026"]
    venues = ["Main Auditorium", "Seminar Hall B", "Innovation Lab"]
    notices = [
        f"The {club} announces the {event} on {date} in {venue}."
        for club in clubs
        for event in events
        for date in dates[:2]
        for venue in venues[:3]
    ]
    notices = (notices * 10)[:N_PROMPTS]
    prompts = [template.format(notice=notice) for notice in notices]

    total_new_tokens = 0
    start = time.time()
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i : i + BATCH_SIZE]
        enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        total_new_tokens += int(out.shape[1] - enc["input_ids"].shape[1]) * len(batch)
    elapsed = time.time() - start

    print(f"{len(prompts)} prompts in {elapsed:.1f}s")
    print(f"~{total_new_tokens / elapsed:.0f} generated tokens/s (approximate)")
    print(f"{len(prompts) / elapsed:.1f} prompts/s")
    return (BATCH_SIZE,)


@app.cell
def _(BATCH_SIZE, MODEL_ID, torch):
    import json
    import platform
    import subprocess
    from pathlib import Path

    import transformers
    from huggingface_hub import HfApi

    revision = HfApi().model_info(MODEL_ID).sha

    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip() or "unknown"

    record = {
        "model_id": MODEL_ID,
        "revision": revision,
        "serving_library": "transformers",
        "serving_library_version": transformers.__version__,
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "quantization": "none",
        "os": platform.platform(),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "driver_version": driver,
        "batch_size": BATCH_SIZE,
    }

    print(json.dumps(record, indent=2))

    destination = Path("runs/day1/environment.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {destination.resolve()}")
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Fill the pre-registration

        Copy the JSON above into `config/experiment.json`:

        - `models.<role>.revision` ← `revision` (40-character commit hash)
        - `environment.serving_library_version` ← `serving_library_version`
        - `environment.torch_version` ← `torch_version`
        - `environment.os` ← `os` (replaces the provisional placeholder)
        - `environment.gpu_name`, `cuda_version`, `driver_version` (confirm or
          correct the provisional values already in the config)
        - `environment.batch_size` ← `batch_size`

        Then fill the matching `TO_FILL` markers in `analysis_plan.md`
        Section 1, mirror the same lines into `project_plan.md`, and re-run
        `bash scripts/hash_frozen.sh` on your machine (update Section 10 in
        both files). Revision 2 requires both files to agree.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Persistence: Google Drive via rclone

        molab has no durable local disk between sessions, so `runs/` must be
        copied out. Revision 2 uses `scripts/sync_runs.sh` (rclone `copy`,
        checksum, never deletes the append-only cache):

        1. On your laptop run `rclone authorize "drive"` and keep the token
           out of the notebook and out of the repo.
        2. In the molab terminal (not a notebook cell), run `rclone config`,
           choose `n` for a new remote named `gdrive`, `drive` as the storage
           type, and paste the token.
        3. Test from this notebook with `DRY_RUN=1` first, then without it.
        4. At session start restore with
           `rclone copy gdrive:promptrepair . --checksum`.

        Native Drive integration in molab is unconfirmed; the plan does not
        rely on it.
        """
    )
    return


@app.cell
def _():
    import os
    import subprocess
    from pathlib import Path

    check = subprocess.run(["rclone", "version"], capture_output=True, text=True)
    script = Path("scripts/sync_runs.sh")
    print(check.stdout.splitlines()[0] if check.returncode == 0 else check.stderr.strip())

    if not script.exists():
        print(f"`{script}` not found in {Path.cwd()}.")
        print("Upload the repository (or at least scripts/) to molab, then re-run.")
        print("Inline equivalent once the `gdrive` remote exists:")
        print("  rclone copy runs gdrive:promptrepair/runs --checksum --progress")
    else:
        env = {**os.environ, "DRY_RUN": "1"}
        sync = subprocess.run(
            ["bash", str(script)], capture_output=True, text=True, env=env
        )
        print(sync.stdout)
        print(sync.stderr)
        print(f"exit={sync.returncode}")
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Day-1 exit checklist

        - [ ] `runs/day1/environment.json` written (revision, versions, GPU,
              CUDA, driver, batch size)
        - [ ] Values copied into `config/experiment.json` and both plan files;
              `bash scripts/hash_frozen.sh` re-run and Section 10 updated
        - [ ] One model downloads and generates
        - [ ] Measured tokens/s and prompts/s written down
        - [ ] Drive round-trip confirmed with `DRY_RUN=1` then a real copy

        **Reminder:** this notebook needs a GPU; run it on molab, not locally.
        """
    )
    return


if __name__ == "__main__":
    app.run()
