# PromptRepair: Five-Day Completion Plan

## Current execution status (29 October 2026)

The D6 single-setting run has reached final evaluation and report generation.
The D6 screening, repair/selection, and 40-notice final-test artifacts are in
`runs/`. The report is `runs/FINAL_REPORT.md`; the paired statistics are in
`runs/d6_statistics.json`.

The original naturalistic prompt's development headroom gate is complete:
20/20 development outputs failed strict scoring because each was wrapped in a
Markdown JSON code fence, so the original-prompt arm qualifies for inclusion.
Its final test is now complete: 0/40 strict accuracy; 38 outputs were fenced
JSON and two were valid JSON with null-policy violations. The restricted
baseline scored 18/20 on development with valid JSON and schema on all 20; its
final-test result was 35/40 (87.5%), with valid JSON/schema on all 40. It ran
after other final-test results were observed, without prompt changes.
Exploratory L-Probe was not run. Only
D6 survived screening, so cross-setting confirmatory analyses and Holm
correction are unavailable; reported comparisons are descriptive. Molab used
Transformers 5.21.0 although the frozen plan specified 5.20.2. See
`runs/ERROR_ANALYSIS.md` for the remaining error cases and `CHAT_HANDOFF.md`
for run details and limitations. Any further prompt optimization should use a
separately collected and reviewed exploratory set; do not tune against the
observed selection or final-test cases.

## 1. What this project is

PromptRepair is a controlled research study about repairing faulty prompts for
structured information extraction.

The model receives a fictional college event notice and must return exactly one
JSON object:

```json
{"event": "...", "date": "...", "venue": "..."}
```

The study deliberately introduces defects into a correct prompt and compares
different repair strategies:

- rewrite the entire prompt;
- edit one local span;
- rewrite one targeted prompt section;
- use rule-based, behavioral, or LLM-based section localization.

The main scientific question is whether targeted repair can recover the
performance of the correct prompt while making fewer and smaller changes than
a full rewrite.

This is an inference-and-evaluation project, not a model-training project. The
five-day schedule should prioritize completing the experiment and producing
auditable results over fine-tuning a model.

## 2. Completion target

By the end of Day 5, the repository should contain:

1. a frozen task contract, prompts, dataset, labels, and hashes;
2. a reproducible local inference record;
3. completed baseline, defect-screening, repair, selection, and final-test runs;
4. raw prompts and model responses stored in resumable run artifacts;
5. statistical analysis and figures/tables;
6. a concise report describing results, limitations, and deviations;
7. a clean test and validation run, or a documented explanation for any
   environment-only failure.

The five-day version should use one model, one greedy decoding configuration,
and the existing 60-notice dataset. Do not add a second model, generated test
set, hosted API, or fine-tuning unless the core experiment is already complete.

## 3. Recommended technical solution

### Model

Use the already selected `Qwen/Qwen2.5-14B-Instruct` model with:

- Transformers;
- `bfloat16`;
- no quantization for the primary run;
- greedy decoding (`temperature=0`, `do_sample=False`);
- fixed batch size chosen from the GPU smoke test;
- pinned Hugging Face revision.

The model is large enough for the extraction and editing task, and using the
same model for extraction and editing keeps the comparison simple. A second
model is a stretch goal only.

### Training versus inference

No training or fine-tuning is recommended in the five-day schedule.

The prompt-repair research question concerns prompt changes, so fine-tuning
would introduce an additional intervention and consume time on dataset
preparation, checkpoint management, validation, and ablation design. The
appropriate use of compute is batched inference and repeated evaluation.

Estimated model time:

- environment/model loading: approximately 10–30 minutes, depending on model
  download and cache state;
- development screening and baselines: approximately 15–45 minutes;
- editing and selection generations: approximately 30–120 minutes;
- final test generation: approximately 10–30 minutes;
- analysis and report generation: generally CPU-bound and much faster than
  model inference.

The exact time must be measured and recorded in the run manifest. The existing
throughput estimate of 19.7 prompts/second is useful for planning but is not a
substitute for the actual run record.

### Dataset

Use the current 60 handwritten fictional notices:

- feedback: 10 notices;
- selection: 10 notices;
- final test: 40 notices;
- 26 notice families, isolated across splits.

Keep the six existing categories: complete, absent venue/date, deadline
distractor, explicit correction, and unresolved conflict. Do not enlarge the
dataset during the five-day execution window. Dataset expansion would require
new labels, family checks, review, hashing, and potentially a revised analysis
plan.

### Storage

Use an append-only run directory under `runs/<run_id>/` with:

- `manifest.json`: model, revision, library, GPU, CUDA, seed, batch size,
  timestamps, and git commit;
- `jobs.jsonl`: every requested generation and its cache key;
- `generations.jsonl`: raw successful and failed generation records;
- `scores.jsonl`: parsed response scores and issue codes;
- `feedback_packets.json`: exact editor inputs;
- `prompts.jsonl`: every prompt version used;
- `results.json`: summarized metrics;
- `deviations.md`: any deviation from the preregistered plan.

The cache must be append-only and keyed by model identity, revision, serving
library version, quantization, rendered prompt, notice, and seed. Sync the run
directory to persistent storage after every major stage. Do not store tokens,
credentials, or private data in the repository.

## 4. Work split: owner versus AI-assisted agents

### You must decide or perform

These actions require owner approval or human accountability:

- approve the final task contract and dataset labels;
- perform or approve the second-pass label review;
- provide the GPU/molab environment record;
- create and test the persistent Drive/rclone destination;
- decide the patch-all-sections/template inconsistency;
- approve the preregistration freeze and hash block;
- review admissibility decisions without looking at final test outcomes;
- approve deviations from the plan;
- write or approve the final interpretation and limitations.

### AI-assisted agents can perform

Agents can safely handle implementation and analysis work:

- complete missing runner, storage, and reporting functions;
- implement the L-LLM adapter interface;
- add deterministic mock adapters and integration tests;
- validate all prompt and dataset invariants;
- run development screening and create feedback packets;
- generate and score model outputs;
- resume failed or interrupted jobs from cache;
- calculate bootstrap and signed-rank statistics;
- produce tables, figures, and draft report text;
- audit manifests, hashes, file completeness, and reproducibility;
- identify code defects and propose patches.

Agents must not silently change labels, frozen prompts, acceptance rules,
exclusion rules, or the statistical plan. Any such change must be reported as a
deviation and approved by you.

## 5. Five-day execution plan

### Day 1 — Freeze the experiment and make the pipeline runnable

Owner tasks:

1. Run the GPU environment check in the molab/GPU environment.
2. Record the exact model commit, Transformers version, Torch version, OS,
   GPU, CUDA, driver, and usable batch size.
3. Perform the second-pass label review on all 60 notices.
4. Resolve the patch-all-sections decision:
   - recommended: keep the all-four-section arm and use the existing
     `section_editor.txt`; or
   - explicitly restrict that arm to the two permitted sections and record the
     deviation.
5. Confirm the persistent storage destination and test a write/read after
   restarting the session.

AI-assisted work:

1. Fix the known unit-test failure and make test execution work in the project
   workspace.
2. Implement `save_run_artifacts()` and `run_sweep()`.
3. Implement final result serialization and report helpers.
4. Add a smoke-test mode using a fake adapter.
5. Validate all 60 records, prompt sections, defect injections, hashes, and
   manifest fields.
6. Produce a freeze checklist showing all remaining `TO_FILL`, `TBD`, and
   unresolved placeholders.

Day 1 exit condition:

- all owner decisions are recorded;
- dataset status is `frozen` or the exact deviation is documented;
- the pipeline completes a fake end-to-end run;
- the actual model environment is recorded;
- frozen artifacts have a new hash block.

### Day 2 — Baselines and defect screening

AI-assisted work:

1. Run the oracle prompt on the 20 development notices.
2. Run the original prompt on the same notices.
3. Run all eight defective prompts on development notices.
4. Score every output using the strict evaluator.
5. Apply the preregistered defect-retention thresholds.
6. Check the oracle ceiling and original-prompt headroom.
7. Build deterministic feedback packets for each retained setting.
8. Save all raw generations, parsed scores, and screening decisions.

Owner review:

- inspect the screening table;
- confirm that no final-test output was used in screening;
- review any transport failure, malformed response, or unexpected defect
  behavior;
- approve the retained settings before editing begins.

Day 2 exit condition:

- every retained setting has a complete development record;
- dropped settings have explicit reasons;
- all feedback packets are reproducible from saved inputs;
- no test labels have been used to choose a repair.

### Day 3 — Prompt repair and selection

AI-assisted work:

1. Run the rewrite branch for each retained setting.
2. Run the targeted span-edit branch.
3. Run the four-section patch branch.
4. Run L-Rule localization.
5. Run L-Behav localization using neutralization outputs.
6. Run L-LLM localization if the adapter is ready; otherwise mark it as an
   explicit unavailable arm and do not call it a completed primary comparison.
7. Apply the two-attempt limit and acceptance rule.
8. Score candidates on the selection set only.
9. Run the admissibility checks and preserve rejected proposals with reasons.
10. Save the final selected prompt for every arm and setting.

Owner review:

- conduct or delegate a blind admissibility review;
- confirm that repairs do not include case-specific answers or test examples;
- confirm that rejected candidates remain rejected;
- approve the final prompt list before test evaluation.

Day 3 exit condition:

- each arm has a selected prompt or a documented no-candidate result;
- each selected prompt is linked to its feedback, selection score, and repair
  history;
- all prompts pass word-limit and schema-preservation checks.

### Day 4 — Final test evaluation

AI-assisted work:

1. Freeze the selected prompt list before reading test labels.
2. Run each final prompt exactly once on all 40 test notices.
3. Score outputs using the test labels only after generation is complete.
4. Compute strict record accuracy and field accuracy.
5. Compute recovery and regression against the original baseline.
6. Compute restoration ratios against the oracle.
7. Sync all raw artifacts to persistent storage.
8. Verify that every expected job has exactly one successful or explicitly
   failed record.

Owner review:

- confirm that the final prompt list was not changed after test scoring;
- inspect the job completeness report;
- approve any exclusion or rerun, with a deviation entry.

Day 4 exit condition:

- all final-test generations and scores are stored;
- there are no unexplained missing jobs;
- the final prompt versions are immutable and hashable.

### Day 5 — Statistics, report, and release

AI-assisted work:

1. Produce per-setting and aggregate accuracy tables.
2. Run family-clustered paired bootstrap confidence intervals.
3. Run the across-setting exact signed-rank analyses specified in the plan.
4. Apply Holm correction only to the preregistered confirmatory comparisons.
5. Produce recovery/regression and restoration-ratio figures.
6. Produce localization rank and mean-reciprocal-rank summaries if the
   localization arms completed.
7. Generate a deviation report and a complete artifact inventory.
8. Run the full test suite and data validator.
9. Draft the final report and executive summary.

Owner tasks:

- review every result table for interpretation errors;
- distinguish confirmatory results from exploratory results;
- document limitations, especially single-model scope and small number of
   settings/families;
- approve the final repository snapshot and report.

Day 5 exit condition:

- report, tables, figures, raw results, manifests, and hashes are complete;
- all deviations are disclosed;
- the repository can be rerun from the documented commands.

## 6. Implementation order for the AI agents

The agents should work in this order because later stages depend on earlier
ones:

1. repair the test-environment issue and known failing assertion;
2. implement storage and run serialization;
3. implement `run_sweep()` around the existing adapter and cache;
4. implement scoring/report aggregation;
5. implement the L-LLM adapter contract;
6. add fake-adapter integration tests;
7. run a full local smoke test;
8. run real development generations;
9. run repair generations;
10. run final-test generations;
11. produce analysis and report artifacts.

No agent should begin real model sweeps before the fake end-to-end run passes.

## 7. Go/no-go rules

Stop and ask for owner input if any of these occur:

- labels disagree during second-pass review;
- the oracle fails the development gate and changing it would alter the frozen
  prompt;
- the GPU cannot fit the model under the registered configuration;
- the model revision or serving library cannot be pinned;
- persistent storage cannot be verified;
- the patch-all-sections arm cannot be defined consistently;
- a repair uses test examples or test labels;
- a generation cache has conflicting records for the same cache key.

Use a smaller batch size or bounded retry for an inference failure. Do not
silently change the model, quantization, decoding strategy, dataset, or
acceptance rule to make the experiment pass.

## 8. Minimal commands for the final workflow

```powershell
python -m pytest -q
python scripts/validate_data.py
```

Then, in the GPU environment:

```powershell
python notebooks/day1_setup.py
```

The real runner should then be invoked through the project’s documented
experiment entry point once `run_sweep()` is implemented. Every run must save
its manifest and cache under `runs/<run_id>/` and sync that directory before
moving to the next day.

## 9. Practical definition of success

The project is complete in five days if it produces a defensible answer to the
research question, even if some exploratory arms are unavailable. A smaller,
fully reproducible one-model study is preferable to a larger study with
unfrozen labels, missing raw outputs, or undocumented prompt changes.

