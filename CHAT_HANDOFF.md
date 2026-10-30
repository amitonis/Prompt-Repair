# PromptRepair — Chat Handoff Summary

Updated: 30 October 2026

## Purpose of the project

PromptRepair is a research project testing whether repairing only the faulty
part of a prompt works as well as rewriting the entire prompt.

The model reads fictional college event notices and must return exactly one JSON
object with these keys:

```json
{"event": "...", "date": "...", "venue": "..."}
```

The project compares full prompt rewrites, local span edits, targeted section
rewrites, and prompt-section localization methods.

## Project structure

- `task_contract.md` — frozen extraction and output requirements.
- `data/` — 60 fictional notices and labels.
- `prompts/oracle.json` — correct oracle prompt.
- `prompts/original.json` — vague baseline prompt.
- `prompts/defects/` — eight controlled prompt defects.
- `src/evaluator.py` — strict JSON/schema/field scorer.
- `src/patcher.py` — validates safe prompt edits.
- `src/optimizer.py` — rewrite, span-edit, and section-patch branches.
- `src/inference.py` — local Transformers model adapter.
- `src/experiment_runner.py` — caching, generation records, screening, and
  manifests.
- `src/storage.py` — dataset loading and run artifact persistence.
- `src/stats.py` — family-clustered paired bootstrap.
- `tests/` — unit tests and mocked pipeline tests.
- `notebooks/day1_setup.py` — Molab GPU setup notebook.
- `scripts/validate_data.py` — dataset validation.
- `PROJECT_COMPLETION_PLAN.md` — five-day completion plan.
- `PROJECT_OVERVIEW.md` — short explanation of the project and workings.

## What was completed in this chat

### Planning and documentation

Created:

- `PROJECT_COMPLETION_PLAN.md`
- `PROJECT_OVERVIEW.md`
- this file, `CHAT_HANDOFF.md`

The completion plan defines the five-day schedule, owner responsibilities,
AI-assisted work, dataset/model/storage decisions, go/no-go rules, and release
criteria.

### Local code changes

- Fixed the failing optimizer test caused by checking an integer with string
  membership.
- Implemented `save_run_artifacts()` in `src/storage.py`.
- Added `scripts/record_environment.py` as a fallback for Molab environments
  where the notebook cell or terminal workflow is inconvenient.
- Confirmed the local project test suite passes.

### Dataset review

The second-pass label review was completed at the owner's request.

Verified:

- 60 records;
- 26 families;
- correct 10/10/40 feedback/selection/test split;
- no family crosses splits;
- all labels satisfy the output schema;
- all non-null values occur verbatim in the corresponding notice;
- null date and venue policies are consistent;
- all category counts match the manifest.

`data/split_manifest.json` was changed from `review_pending` to `frozen`.
The review was recorded in `project_plan.md`.

### Runtime environment

The user successfully ran the Molab GPU benchmark. Frozen environment values:

```json
{
  "model_id": "Qwen/Qwen2.5-14B-Instruct",
  "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
  "serving_library": "transformers",
  "serving_library_version": "5.20.2",
  "torch_version": "2.11.0+cu130",
  "dtype": "bfloat16",
  "quantization": "none",
  "os": "Linux-4.19.0-gvisor-x86_64-with-glibc2.41",
  "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "cuda_version": "13.0",
  "driver_version": "595.71.05",
  "batch_size": 16,
  "measured_throughput_prompts_per_s": 6.1
}
```

This record was added to:

- `config/experiment.json`;
- `analysis_plan.md`;
- `project_plan.md`;
- `runs/day1/environment.json`.

### Verification results

The local test suite currently passes:

```text
142 passed
```

The data validator passes with:

```text
PASSED: all dataset checks
```

The Molab validation also passed. Its final `SystemExit: 0` traceback was a
normal script exit that Marimo displayed as an exception; it was not a data
failure.

### Latest model baseline

The first model baseline using plain prompt transport was invalid for the
intended Qwen instruct interface and scored 25% strict accuracy, with mostly
malformed or repeated outputs. The adapter was corrected to use the Qwen
tokenizer chat template while preserving the rendered task prompt.

The corrected oracle development baseline passed:

```text
20 notices
strict_accuracy: 0.90 (18/20)
elapsed_s: 3.60
```

The two remaining errors were `SEL-005` and `SEL-006`, where the model included
the organizer phrase in the event name. No test notices were used. This is
recorded as pre-freeze deviation D-01 in `analysis_plan.md`.

### Latest execution status

The corrected baseline is now the valid development result:

```text
oracle strict accuracy: 0.90 (18/20)
oracle gate: PASSED (required >= 0.85)
```

The inference adapter now uses Qwen's tokenizer chat template with
`add_generation_prompt=True` and passes `eos_token_id`. This fixed the earlier
plain-prompt transport problem that produced malformed, repeated, and
prompt-continuation outputs.

The local project test suite now passes:

```text
143 passed
```

The Molab notebook was expanded with a new button:

```text
Run all eight defect-screening evaluations
```

The updated files are:

- `molab_one_file_runner.py`
- `promptrepair_molab_bundle.zip`
- `scripts/build_molab_bundle.py`

The ZIP must be uploaded again after notebook or source updates. The notebook
recreates the project on every Molab reset.

### GPU usage rule

GPU is not needed for ZIP extraction, dataset validation, automated tests,
statistics, tables, or report writing.

GPU is needed for the model smoke test, oracle baseline, defect screening,
editor/repair calls, selection evaluation, and final 40-notice test. Reattach
the GPU after every Molab reset before loading the model.

## Molab workflow discovered

Molab is a cloud-hosted Marimo notebook environment. It provides GPU compute,
but the user's interface only allowed individual file uploads rather than a
project folder.

The uploadable archive is generated by `scripts/build_molab_bundle.py` and
includes current source, frozen inputs, and the saved run/report JSON files.
The script excludes Python cache files and unused generated-test duplicates.

- `promptrepair_molab_bundle.zip`

It contains the source, configuration, frozen data, prompts, scripts, tests,
documentation, and saved run artifacts. Upload it with the current
`molab_one_file_runner.py` after rebuilding it locally.

The user uploaded the ZIP and extracted it in a Molab cell with:

```python
import zipfile as zipfile_module
from pathlib import Path as FilePath

bundle_file = FilePath("promptrepair_molab_bundle.zip")
project_folder = FilePath("promptrepair")

with zipfile_module.ZipFile(bundle_file) as bundle_archive:
    bundle_archive.extractall(project_folder)
```

Because Marimo cells are reactive, variables should not be redefined in
multiple cells. Use unique variable names such as `tests_project_root` instead
of reusing `Path` or `validation_root`.

## Latest execution status: defect screening complete

The user ran all eight preregistered defect screens on the 20 development
notices (20 notices per defect). The screening run took about 21.49 seconds.
Results, compared with the 90% oracle development baseline (18/20):

| Defect | Accuracy | Change vs oracle | Flips | Retained? | Main failure result |
| --- | ---: | ---: | ---: | --- | --- |
| D1 | 90% (18/20) | 0 points | 0/10 | No | 2 event mismatches |
| D2 | 85% (17/20) | -5 points | 0/4 | No | 2 event mismatches; 1 null-policy violation |
| D3 | 90% (18/20) | 0 points | 0/2 | No | 2 event mismatches |
| D4 | 90% (18/20) | 0 points | 0/10 | No | 2 event mismatches |
| D5 | 90% (18/20) | 0 points | 0/20 | No | 2 event mismatches |
| D6 | 0% (0/20) | -90 points | 18/20 | **Yes** | 20 schema-key failures; 2 event mismatches |
| D7 | 90% (18/20) | 0 points | 0/20 | No | 2 event mismatches |
| D8 | 90% (18/20) | 0 points | 0/20 | No | 2 event mismatches |

Only D6, the defect that drops required output keys/schema, passed its
preregistered retention rule. The other seven defects did not meet their
screening thresholds. The two original oracle errors persist in most arms;
D2 adds one null-policy error, while D6 causes widespread schema failures.

### Interpretation and scope

Continue the experiment with D6 as the sole retained defect. The planned
multi-setting comparisons cannot be treated as confirmatory because only one
defect setting survived screening. Report the other seven as screened out and
describe any D6 repair-arm comparisons with that limited scope. Do not broaden
claims to prompt defects generally.

GPU inference was used for screening, D6 repair/selection, and final evaluation.
Data checks, aggregation, and report generation ran without the GPU.

## D6 pipeline implementation

The D6 repair/selection orchestration was added locally. The Molab run used the
equivalent code in the user-pasted notebook cell:

- `src/pipeline.py::run_d6_repair_sweep()` runs D6 feedback evaluation,
  feedback-packet construction, rewrite, span, targeted-section editing,
  localizer lookups, and selection-set acceptance.
- `src/pipeline.py::evaluate_locked_arms()` evaluates locked prompts on the
  test split only after selection is complete.
- `src/experiment_runner.py::run_sweep()` now delegates to the D6 pipeline.
- `src/reporting.py::build_report()` now builds a JSON-compatible final report
  table and marks the one-setting analysis as descriptive-only.

### Molab D6 repair/selection result received

The user pasted `d6_repair_selection.json`, which has been saved locally under
`runs/`. It completed in about 74.28 seconds. The D6 defective baseline scored
0/10 on both feedback and selection because the required `venue` key was
missing. On selection, rewrite and span each reached 8/10 and were accepted.
L-Rule, L-LLM and L-Behav all ranked `output_format` first; their selected
section patch reached 8/10 and was accepted. The oracle-section patch also
reached 8/10. The four random-section patch candidates scored 0/10, 1/10,
0/10, and 8/10 (instruction, context, constraints, output_format).

### Final test and report completed

The final-test artifact is saved as `runs/d6_final_test.json` (40 notices,
14 families; 23.21 seconds). The report and computed statistics are
`runs/FINAL_REPORT.md` and `runs/d6_statistics.json`. The reproducible report
generator is `scripts/generate_final_report.py`.

- Defective D6 prompt: 0/40 (0%); valid JSON but missing required schema key.
- Rewrite, span, L-Rule, L-LLM, L-Behav, oracle-section and oracle: each 35/40
  (87.5%), with valid schema on all 40.
- Random section candidates: instruction 0/40, context 11/40, constraints
  0/40, output_format 35/40. Mean: 28.75%.
- P1 (L-LLM minus rewrite): 0 percentage-point difference; 95% family
  bootstrap CI [0, 0]; no discordant notices.
- P2 (oracle-section minus random-section mean): +58.75 points; 95% family
  bootstrap CI [+46.51, +69.87] points; exact family sign-flip p=0.000244.
  The bootstrap had zero nonpositive draws out of 10,000, reported as
  p < 0.0001 for its finite Monte Carlo estimate.
- Oracle sensitivity excluding TEST-001 to TEST-004: 34/36 (94.44%).
- Runtime deviation: Transformers 5.21.0 was used; the pinned plan specified
  5.20.2.

Only D6 survived screening, so the preregistered across-setting confirmatory
tests and Holm correction were not performed. All comparisons in this single
setting are descriptive. The D6 final test and primary comparisons are
complete. The original-prompt gate/final test and restricted-baseline
development/final test are also complete. The restricted baseline scored
18/20 on development and 35/40 on the final test. The L-Probe is an optional
exploratory arm and was not run.

### Completed: original-prompt development headroom gate

The user supplied the completed `runs/original_development.json` result after
restarting Molab and running the gate. All 20 feedback + selection notices
failed strict evaluation in 3.58 seconds, so the original-prompt arm passes
the preregistered inclusion threshold of at least three failures. Every raw
response contains an otherwise plausible JSON object wrapped in a Markdown
`json` code fence; the strict parser therefore marked each as malformed JSON.
This gate result does not include a final-test evaluation of the original arm.

The report generator now records the gate metrics and its failure mode. The
user subsequently ran the original prompt on the locked 40-notice test set.
The result is saved as `runs/original_final_test.json`: 0/40 strict accuracy,
with only 2/40 strictly valid JSON responses. The other 38 were fenced JSON;
the two parseable responses violated the null policy for unresolved date and
venue conflicts. `scripts/generate_final_report.py` now includes these metrics
in the final arm table. The restricted baseline development artifact is saved
as `runs/restricted_baseline_development.json`: 18/20 strict accuracy (90%),
with 20/20 valid JSON and schema. The preregistered arm was executed after
other test results were observed; no prompt changes were made. Its final-test
artifact is saved as `runs/restricted_baseline_final_test.json`: 35/40 strict
accuracy (87.5%), with 40/40 valid JSON and schema. The report and Molab bundle
were refreshed. Exploratory L-Probe remains unrun.

The case-level analysis is saved as `runs/ERROR_ANALYSIS.md` and generated by
`scripts/generate_error_analysis.py`. The rewrite, oracle, and restricted
baseline share five strict errors: three event-name boundary/punctuation cases
and two unresolved-date cases where the model selected one date instead of
returning null. The potential punctuation-label ambiguity is documented but
the frozen labels and scores were not changed. The restricted development run
was 10/10 on feedback and 8/10 on selection; both errors were selection cases,
so they must not be reused to tune the primary prompt under the frozen
procedure. A future error-focused extension needs a separately collected and
reviewed set.

Separate, unscored exploratory drafts are stored under
`data/exploratory_draft_01/` and `data/exploratory_draft_02/`. The advisory
second-pass log and review notes are preserved in draft 01. After the owner
chose to rename EB-004 and update its event/venue gold/evidence, v2 was derived
from draft 01; UC-003 also no longer calls the flyer "current". Draft 01 stays
unchanged. The owner reviewed the v2 decisions in chat and authorized a
separate exploratory run; the review record is in
`data/exploratory_draft_02/human_review.json`. V2 is now included in the Molab
bundle, but its results must remain raw counts, post hoc, exploratory, and
separate from the primary report. Report EB-005 separately for its punctuation
convention; note the five-date/one-venue conflict mix and explicit no-priority
cues.

The user ran the v2 comparison on Molab and pasted the result on 30 October
2026. The archival summary is `runs/exploratory_v2_comparison.json`; the
standalone result write-up is `runs/EXPLORATORY_V2_REPORT.md`. The oracle
baseline scored 11/12 and the boundary/conflict candidate 10/12; both had
12/12 valid JSON and schema. Both missed EB-005 under the convention that
excludes terminal punctuation. The candidate additionally returned “The Film
Poster Workshop” in UC-003, while correctly returning a null venue. Keep this
post-hoc result out of primary scores and conclusions. Full raw generations
remain in the Molab output; the local JSON is a compact archival summary.

Exploratory draft 03 is a fresh 12-case set under `data/exploratory_draft_03/`.
The owner accepted every proposed label unchanged on 30 October 2026, including
the question mark in EXP3-004. Decisions and separate exploratory run
authorization are recorded in `human_review.json`; reviewed labels are saved
in `labels_reviewed.jsonl`. `scripts/run_exploratory_v3.py` compares the
baseline with the frozen v2 candidate on this set. Keep its raw-count results
separate from the primary study and do not tune the candidate on v3 outcomes.

The Molab v3 run completed on all 12 cases. Both the oracle baseline and frozen
v2 candidate were correct on 12/12, with valid JSON and schema on 12/12 and no
incorrect cases. The local compact result is
`runs/exploratory_v3_comparison.json`; the separate write-up is
`runs/EXPLORATORY_V3_REPORT.md`. Input, label, and prompt hashes match the
project files. Full raw generations remain in Molab.

Research-informed recommendations are recorded in `RESEARCH_IMPROVEMENT_PLAN.md`.
Priorities: clarify future annotation rules for title boundaries, add field-level
error diagnostics without changing frozen scores, and design a larger blind
evaluation sized by a paired minimum-detectable-effect calculation before using
automatic prompt optimizers.

The first offline diagnostic step is complete. `scripts/generate_field_diagnostics.py`
creates `runs/FIELD_DIAGNOSTICS.md` and `runs/field_diagnostics.json` from the
saved final-test artifacts. It reports exact field matches separately from
record accuracy and schema validity. The leading prompt arms each have event
37/40, date 38/40, and venue 40/40 exact matches; the five record errors remain
TEST-001, TEST-003, TEST-004, TEST-037, and TEST-038. This analysis does not
change any frozen score or label.

Tier A reviewer prompt-repair prototype is in `src/reviewer_tool.py`, with CLI
entry point `scripts/repair_prompt.py`. It accepts sectioned JSON or plain-text
instructions, uses only the fixed feedback and selection development sets,
compares a rule-localized section patch with a full rewrite, and saves before/
after prompts, diffs, rejected candidates, and reports. Three demo prompt files
are under `examples/reviewer_prompts/`. CLI `--help` works locally. Tier B
arbitrary-task evaluation remains unimplemented.

All three Tier A demos have now run in Molab on feedback/selection development
data. `runs/reviewer_prompt_demo_summary.json` and
`runs/REVIEWER_PROMPT_DEMO_REPORT.md` record the pasted summaries. Selection
counts were 0/10 before each repair; targeted/rewrite results were 7/10 and
5/10 for the vague prompt, 8/10 and 10/10 for the missing venue key, and 0/10
(rejected, no change) and 9/10 for the wrong conflict rule. These are demos,
not general performance claims. The targeted rejection trace for the third
demo trace was inspected. L-Rule ranked `output_format` first and `constraints`
second. The format patch was valid but left all 10 selection outputs malformed,
so the selection acceptance rule rejected it. The targeted branch previously
proposed only the top-ranked section; it now scores both highest-ranked sections
and lets the unchanged acceptance rule choose an edit. The constraints rewrite
exceeded the 60-word edit cap. V3 repeated the same 79-word replacement on both
attempts. V4 added validator feedback to attempt 2, but the editor still repeated
the same proposal; targeted mode remained rejected with no score change. The
earlier `both` run's full rewrite succeeded at 9/10 selection records. Results
are archived in `runs/reviewer_prompt_retry_v3_summary.json` and
`runs/reviewer_prompt_retry_v4_summary.json`. A final Molab `both` run reproduced
the same result: targeted rejected at 0/10; rewrite accepted at 9/10 with 10/10
valid JSON and schema. Summary: `runs/reviewer_prompt_molab_final_summary.json`.
Stop repeating this targeted demo; use the failure as a documented limitation
while finishing Tier A packaging.

## Broader project direction

The user wants to move beyond the fixed notice task and asked about one 8–10-hour
training run. `RESEARCH_BROADER_PROMPT_REPAIR_8H.md` reviews prompt optimization,
QLoRA, broad task datasets, and proposes a staged run. Recommendation: first
build cross-task repair records and a task-metric interface, then use a fixed
model for a budgeted multi-task optimization study. Train a QLoRA adapter only
after repair examples and task-family held-out splits are ready. The Molab GPU
previously reported 94.97 GiB total memory, but throughput and availability
must be checked at the next run. A fresh standalone bootstrap notebook,
`molab_broad_repair_runner.py`, now uploads and hash-verifies the ZIP in a
separate runtime directory and reports GPU, training-package, and dataset
readiness. It does not load weights or train.

## Constraints and decisions

- Do not fine-tune a model for the five-day core project.
- Use one pinned model and greedy decoding.
- Keep the 60-notice dataset unchanged after freezing.
- Do not use final-test labels to select prompts.
- Do not silently change prompts, labels, model, decoding, or acceptance rules.
- Record every deviation.
- Prefer a smaller complete reproducible run over a larger incomplete run.

