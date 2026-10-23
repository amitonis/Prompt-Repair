# PromptRepair

Targeted prompt correction for structured text extraction: does repairing one
faulty prompt section work as well as or better than rewriting the entire
prompt?

Full methodology: `PromptRepair_In_Depth_Explanation_and_Methodology_v2.md`
(includes related work, novelty claims, experimental design, and the 14-day
plan). Pre-registered analysis plan: `analysis_plan.md`.

## Task

Extract `event`, `date`, and `venue` (each a string or JSON null) from
fictional college event notices, under the fixed contract in
`task_contract.md`. Models are open-weight instruct models served locally on a
marimo molab GPU session; no paid API is required.

## Repository layout

```
app.py                      marimo app (thin front end over src/)
notebooks/                  marimo notebooks: setup, sweeps, analysis
analysis_plan.md            pre-registered plan (commit before sweeps)
task_contract.md            frozen task specification
config/                     experiment settings and validation schema
data/                       feedback/selection/test notices and labels
prompts/                    original, oracle, defect catalog, editor templates
src/                        pipeline modules
tests/                      unit and pipeline tests (mocked models)
runs/                       per-run artifacts (not committed)
```

## Implemented so far

- `src/evaluator.py`: strict response scorer + character edit distance
  (reference implementation from the methodology guide)
- `src/prompt_renderer.py`: four-section template rendering
- `src/defects.py`: oracle prompt loader and single-section defect injection
- `src/patcher.py`: patch guard enforcing the targeted-edit rules
- `src/diagnostics.py`: deterministic feedback-packet construction
- `src/localizers.py`: L-Rule localizer (others stubbed until the adapter exists)
- `src/storage.py`: split loading and family-isolation checks
- `src/optimizer.py`, `src/experiment_runner.py`, `src/reporting.py`:
  selection rule and cache key implemented; sweeps pending the inference
  adapter
- `src/inference.py`: adapter interface; local serving wired up on molab

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python -m pytest
```

GPU-dependent steps (model download, serving, sweeps) run in a marimo molab
session; see Section 13.5 of the methodology guide. The Day-1 environment
check lives in `notebooks/day1_setup.py` — mirror it into a molab GPU
session (never paste tokens into a notebook).

## Status

Planning/scaffold stage. No LLM experiments have been run; all results tables
remain blank by design until the evaluation executes.
