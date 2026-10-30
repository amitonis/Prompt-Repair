# PromptRepair

Targeted prompt correction for structured text extraction: does repairing one
faulty prompt section work as well as or better than rewriting the entire
prompt?

- Full methodology: `PromptRepair_In_Depth_Explanation_and_Methodology_v2.md`
- Pre-registered analysis plan: `analysis_plan.md` (frozen plan; deviations are
  documented in the final report)
- Full project plan plus the decision log and one-week schedule:
  `project_plan.md`

## Task

Extract `event`, `date`, and `venue` (each a string or JSON null) from
fictional college event notices, under the fixed contract in
`task_contract.md`. Models are open-weight instruct models served locally on a
marimo molab GPU session; no paid API is required.

## Repository layout

```
app.py                      marimo app (thin front end over src/)
notebooks/                  marimo notebooks: setup, sweeps, analysis
analysis_plan.md            pre-registered analysis plan (authoritative)
project_plan.md             full plan, decision log and schedule
task_contract.md            frozen task specification
config/                     experiment settings and validation schema
data/                       feedback/selection/test notices and labels
prompts/                    original, oracle, defect catalog, editor templates
src/                        pipeline modules
scripts/                    validation, hashing, sign-flip and sync helpers
tests/                      unit and pipeline tests (mocked models)
runs/                       per-run artifacts (git-ignored)
```

## Implemented so far

- `src/evaluator.py`: strict response scorer (strict parse, schema checks,
  whitespace-only comparison) plus character edit distance
- `src/prompt_renderer.py`: four-section template rendering
- `src/defects.py`: oracle prompt loader and single-section defect injection
- `src/patcher.py`: patch guard enforcing the targeted-edit rules
- `src/diagnostics.py`: deterministic feedback-packet construction
- `src/localizers.py`: L-Rule, L-Behav, and L-LLM localization methods;
  L-Probe remains exploratory and was not run
- `src/storage.py`: split loading and family-isolation checks
- `src/optimizer.py`: selection/acceptance rule, editor-JSON parsing, rewrite,
  span-edit, and targeted-section edit branches with patch admissibility
- `src/experiment_runner.py`: cache key, session manifest, manifest diff,
  append-only resume cache, resumable generation driver, and screening
  thresholds; `src/pipeline.py` orchestrates D6 repair/selection and locked-arm
  evaluation
- `src/stats.py`: family-cluster paired bootstrap
- `src/reporting.py`: result aggregation and report construction
- `src/inference.py`: local `transformers` adapter with revision pinning,
  left-padded batched greedy generation, deterministic batch order,
  OOM batch halving and bounded retries; hosted control remains a stub
- `scripts/validate_data.py`: structural checks on the 60 notices
- `scripts/build_molab_bundle.py`: rebuilds the uploadable restart ZIP with
  current sources, frozen inputs, and saved run results
- `scripts/hash_frozen.sh`: SHA-256 of the frozen artifacts
- `scripts/sign_flip.py`: exact sign-flip test over families
- `scripts/sync_runs.sh`: rclone copy of `runs/` to Google Drive

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python -m pytest
python scripts/validate_data.py
```

GPU-dependent steps (model download, serving, sweeps) run in a marimo molab
session; see the methodology guide and `project_plan.md` Section 14. Rebuild
`promptrepair_molab_bundle.zip` with `python scripts/build_molab_bundle.py`
before uploading it alongside `molab_one_file_runner.py`. The Day-1
environment check lives in `notebooks/day1_setup.py`; never paste tokens into
a notebook.

## Status

As of 30 October 2026, the frozen dataset contains 60 notices in a 10/10/40
feedback/selection/test split across 26 families. D6 screening, repair and
selection, and the 40-notice final test have run on Molab. Only D6 survived
screening; within-setting results are descriptive. The results and limitations
are in [`runs/FINAL_REPORT.md`](runs/FINAL_REPORT.md), the remaining error
cases are analyzed in [`runs/ERROR_ANALYSIS.md`](runs/ERROR_ANALYSIS.md), and
the full handoff is in [`CHAT_HANDOFF.md`](CHAT_HANDOFF.md).

The original-prompt development headroom gate passed (20/20 strict failures;
all outputs were wrapped in Markdown JSON fences), so that arm qualified for
inclusion. Its final-test result was 0/40 strict accuracy (38 fenced responses
and two null-policy violations). The restricted baseline development run
scored 18/20 (90%) with valid JSON/schema on all 20; its final-test result was
35/40 (87.5%), with valid JSON/schema on all 40. This arm ran after other
final-test results were observed, without prompt changes. Exploratory L-Probe
was not run. Molab used Transformers 5.21.0
instead of the planned 5.20.2. These limitations are recorded in the report.
Do not use the locked final-test outcomes to select or tune prompts; any
additional optimization should use development data and be labeled
exploratory.

Separate 12-case exploratory comparisons are recorded under `runs/` and
remain outside the primary study. Draft 02 scored the oracle 11/12 and its
boundary/conflict candidate 10/12; draft 03 scored both the oracle and frozen
candidate 12/12. These are post-hoc synthetic results, reported as raw counts
only, and do not establish general prompt improvement.

## Reviewer prompt repair — Tier A prototype

The command-line prototype repairs prompts for this project's fixed college-
notice extraction task. It scores the supplied prompt on the existing 10-case
feedback and 10-case selection development splits, creates a targeted patch
and/or a full rewrite, and accepts a candidate only when selection exact-record
accuracy strictly improves without increasing schema-invalid outputs. It
never loads the frozen test split. The editor uses the development examples;
the report labels these scores as development results.

Prompt files can be JSON with a `sections` object containing `instruction`,
`context`, `constraints`, and `output_format`. Missing sections are filled from
the canonical notice-task prompt, so include all four sections when the
reviewer's original prompt is already sectioned. A plain-text file is treated
as the instruction section and receives the canonical defaults for the other
three sections. This keeps the task contract and output schema explicit.

Example command (run in Molab or another environment with the pinned model and
GPU available):

```powershell
python scripts/repair_prompt.py examples/reviewer_prompts/demo_missing_venue_key.json --mode both --run-id demo-missing-venue
```

Modes are `targeted`, `rewrite`, and `both`. The output directory contains the
before/after prompt JSON, unified diff, rejected-candidate details, generation
cache, and Markdown/JSON report. In `both` mode the targeted patch and full
rewrite are shown side by side; this is descriptive and does not imply the
patch is better.

### One-cell Molab call

Upload and extract the current Molab bundle first. Then call the tool directly:

```python
from pathlib import Path as ReviewerRepairPath
from src.reviewer_tool import run_repair as run_reviewer_prompt_repair

reviewer_prompt_file = (
    ReviewerRepairPath(bundle_runtime_root)
    / "examples" / "reviewer_prompts" / "demo_missing_venue_key.json"
)
reviewer_repair_output = run_reviewer_prompt_repair(
    reviewer_prompt_file,
    mode="both",
    run_id="molab-demo-missing-venue",
    device="cuda",
)
print("Report:", reviewer_repair_output / "report.md")
```

Tier B (arbitrary tasks scored with a reviewer's own input/output examples and
metric) is not implemented yet. Do not use the frozen 40-case test to repair or
score reviewer prompts. The Tier A CLI implementation is a prototype and
remains development-only until its Molab demo runs are saved.
