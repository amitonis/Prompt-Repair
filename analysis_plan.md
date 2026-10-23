# Analysis Plan (Pre-registration Draft)

**Status: DRAFT.** Commit this file unchanged before the first optimization
run (Day 7, Section 9A.7). Do not alter it after seeing test results; record
any deviation as a deviation.

## 1. Settings

- Task: three-field extraction from fictional college notices
  (`task_contract.md`, frozen).
- Starting prompts: oracle prompt (`prompts/oracle.json`) with one injected
  defect from `prompts/defects/` (D1-D8), plus the naturalistic original
  (`prompts/original.json`) as a setting with no known fault location.
- Models: TO_FILL (extractor and editor model IDs + revisions from
  `config/experiment.json`).

## 2. Defect screening

A defect is kept for a given model only if it lowers strict record accuracy on
the 20 development notices by at least **15 points** relative to the oracle
prompt. Screening uses development data only. Dropped defects are reported.

## 3. Arms

Defective (unchanged) | Rewrite | Span edit | Section patch + localizer X
(L-LLM, L-Rule, L-Probe; L-Ablate if time) | Oracle-section patch |
Random-section patch | Oracle prompt | Restricted baseline (constraints and
output_format only).

All editing arms share the same feedback packet, attempt limit (2), size
limits (60-word new text, 240-word prompt), selection rule, and admissibility
review.

## 4. Localizers

Rankings over the four sections with fixed tie-break order: output_format,
constraints, instruction, context. Selection data is never used by a
localizer; only the acceptance rule uses it.

## 5. Metrics

Strict record accuracy, JSON validity, schema validity, field accuracy,
null-policy accuracy, unwarranted filling, recovery rate, regression rate,
top-1 localization accuracy, mean reciprocal rank, restoration ratio (reported
unclipped; N/A on zero denominator), character edit distance, prompt growth,
editor calls, extraction generations, GPU time.

## 6. Statistical analysis

1. Within a setting: paired analysis over notices (exact McNemar or paired
   bootstrap).
2. Across settings: bootstrap or signed-rank with the setting as the unit,
   with confidence intervals.
3. All arms reported, including losers. Small setting counts mean tests are
   descriptive; avoid strong language.

Strict localization (matches the injected section) and fixability
(restoration ratio above 0.8) are reported separately.

## 7. Seeds and determinism

Greedy decoding for the primary comparison; seed 20261014 for random-section
choices and any sampled arms; cache keys per `src/experiment_runner.py`.

## 8. Deviations

None yet.
