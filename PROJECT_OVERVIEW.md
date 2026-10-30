# PromptRepair — Project Overview

## In one sentence

PromptRepair tests whether fixing only the faulty part of an information-
extraction prompt works as well as rewriting the entire prompt.

## The task

The model reads fictional college event notices and extracts three values:

```json
{"event": "event name", "date": "event date or null", "venue": "venue or null"}
```

The answer must be exactly one JSON object with exactly these keys. Values must
be copied from the notice without paraphrasing or date reformatting. Missing or
unresolved values become JSON `null`.

## How the experiment works

1. Start with an oracle prompt that correctly describes the task.
2. Create defective versions by changing one prompt section at a time.
3. Run the oracle, original, and defective prompts on development notices.
4. Keep only defects that create meaningful, pre-defined failures.
5. Give the repair model a controlled feedback packet containing selected
   failing and passing examples.
6. Ask repair strategies to propose new prompts.
7. Accept a repair only when selection accuracy strictly improves and invalid
   outputs do not increase.
8. Run the selected prompts on the untouched final test set.
9. Compare accuracy, recovery, regressions, prompt size, and repair cost.

## Main components

- `task_contract.md`: frozen definition of the extraction task.
- `data/`: feedback, selection, and final-test notices plus labels.
- `prompts/oracle.json`: correct four-section prompt.
- `prompts/original.json`: vague baseline prompt.
- `prompts/defects/`: eight controlled prompt defects.
- `src/evaluator.py`: strict JSON and semantic scorer.
- `src/patcher.py`: guardrails for admissible prompt edits.
- `src/optimizer.py`: rewrite and targeted-repair branches.
- `src/inference.py`: local Transformers model adapter.
- `src/experiment_runner.py`: caching, manifests, retries, and screening.
- `src/stats.py`: family-clustered bootstrap statistics.
- `tests/`: tests for the core logic and mocked inference.
- `runs/`: raw outputs and reproducible experiment artifacts.

## Prompt sections

Every reusable prompt contains four sections:

1. `instruction` — what to extract;
2. `context` — assumptions about the notice;
3. `constraints` — how to interpret dates, venues, nulls, and conflicts;
4. `output_format` — exact JSON requirements.

The notice is inserted at runtime after these sections. This separation lets the
study change a prompt section without changing the input notices.

## Dataset design

The current dataset contains 60 fictional notices:

- 10 feedback notices used to show errors to the editor;
- 10 selection notices used to choose among candidate repairs;
- 40 final-test notices used only for the final evaluation;
- 26 families, kept within one split to reduce leakage.

The categories include complete notices, missing dates or venues, deadline
distractors, explicit corrections, and unresolved conflicts.

## What is trained

Nothing is fine-tuned for the core project. The study uses a fixed pretrained
instruction model and changes only its prompts. This keeps the causal question
focused on prompt repair and makes completion within five days realistic.

The editor and extractor may use the same Qwen 2.5 14B Instruct model. The
primary run uses greedy decoding, fixed batching, and a pinned model revision.

## What is stored

Each run should preserve:

- model and hardware manifest;
- exact rendered prompts;
- notice IDs and cache keys;
- raw model responses;
- parsed scores and error codes;
- repair proposals and rejected proposals;
- selected prompts;
- final metrics and statistical outputs.

The generation cache is append-only. A cache key changes when the model,
revision, serving version, prompt, notice, quantization, or seed changes.

## Current project status

The core evaluator, dataset checks, defect injection, prompt patch validation,
optimizer branches, inference adapter, and many tests are present. The project
still needs the end-to-end sweep runner, persistent run storage, final report
aggregation, L-LLM localization, environment freeze, real GPU runs, and final
analysis.

The five-day plan for completing those items is in
[`PROJECT_COMPLETION_PLAN.md`](PROJECT_COMPLETION_PLAN.md).

## Important limitations

- The primary study uses one model.
- The dataset is small and fictional.
- Notice families are clustered, so notices should not be treated as fully
  independent observations.
- Results are about this task, dataset, model, and prompt design; they do not
  automatically generalize to all prompt-repair problems.
- Exploratory arms must be clearly separated from preregistered comparisons.

