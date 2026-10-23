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

        Purpose: record GPU details, download one open-weight instruct model,
        measure throughput, and verify persistence.

        **Rules**

        - Never paste HF or GitHub tokens into this notebook: molab notebooks
          are public but undiscoverable.
        - Sessions stop after ~12 hours; keep today short.
        - After the session, copy recorded values into `config/experiment.json`
          on your own machine and push to GitHub (molab has no push token).
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
        **Record:** GPU name, VRAM, driver version, CUDA version → copy into
        `config/experiment.json` locally, then commit.

        **Model choice:** pick an **ungated** instruct model. Rough VRAM math:
        bf16 ≈ 2 bytes per parameter (7B ≈ 16 GB, 14B ≈ 28 GB). Examples:
        `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`,
        `mistralai/Mistral-7B-Instruct-v0.3`,
        `microsoft/Phi-3.5-mini-instruct`. Edit MODEL_ID below if needed.
        """
    )
    return


@app.cell
def _():
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

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
    return MODEL_ID, model, tok


@app.cell
def _(mo):
    mo.md(
        """
        ## Throughput benchmark

        ~200 short prompts, batched greedy generation. If it OOMs, halve
        `BATCH_SIZE`. Record **prompts/s** — sweep time ≈ total generations
        ÷ prompts per second. Target: full sweep ≤ 6 hours; if larger, shrink
        per Section 15.2 of the methodology guide (fewer repeats → fewer
        defects → one model).
        """
    )
    return


@app.cell
def _(model, tok):
    import time

    BATCH_SIZE = 32
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
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Persistence checks

        1. **Google Drive:** the cell below writes a smoke file into the
           notebook workspace. Check molab's Drive/files integration (per the
           current molab docs) and confirm the file reaches your Drive. If it
           does, plan to write `runs/RUN_ID/` there after every batch.
        2. **GitHub:** keep the repo public and mirror it into molab. The
           clone cell below needs no token for public repos. Do not push from
           molab (that would need a token in a public notebook); push code from
           your own machine instead.
        """
    )
    return


@app.cell
def _():
    from pathlib import Path

    smoke_dir = Path("runs") / "SMOKE"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_file = smoke_dir / "day1_hello.txt"
    smoke_file.write_text("Day 1 persistence smoke test\n", encoding="utf-8")
    print(f"Wrote {smoke_file.resolve()}")
    return


@app.cell
def _():
    import subprocess

    GITHUB_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO.git"
    result = subprocess.run(
        ["git", "clone", GITHUB_URL], capture_output=True, text=True
    )
    print(result.stdout)
    print(result.stderr)
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Day-1 exit checklist

        - [ ] GPU name / VRAM / driver / CUDA recorded → `config/experiment.json`
        - [ ] One model downloads and generates
        - [ ] Measured tokens/s and prompts/s written down
        - [ ] A file survived via Drive, or fallback confirmed

        **Reminder:** this notebook needs a GPU — run it on molab, not
        locally (unless your own machine has a GPU).
        """
    )
    return


if __name__ == "__main__":
    app.run()
