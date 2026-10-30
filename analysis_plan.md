# Analysis Plan (Pre-registration)

**Status: PRE-FREEZE (becomes FINAL when `prereg-v1` is tagged).** Commit this file unchanged before the first optimization
run (Day 7, Section 9A.7) and tag the commit `prereg-v1`. Do not alter it
after seeing test results; record any change as a deviation (Section 11).

Decisions marked *(proposed)* in the draft were resolved before this commit:
10/10 development split; feedback packet of 5 failing + 2 passing examples;
localizer set L-LLM + L-Rule + L-Behav; Test-G dropped from core;
Qwen2.5-14B-Instruct for both roles (single-model pilot).

## 1. Settings

- Task: three-field extraction (event, date, venue) from fictional college
  notices (`task_contract.md`, frozen; SHA-256 recorded in Section 10).
- Starting prompts: oracle prompt (`prompts/oracle.json`) with one injected
  defect from `prompts/defects/` (D1-D8), plus the naturalistic original
  (`prompts/original.json`) as a setting with no known fault location.
- A **setting** = (model, starting prompt). Results are reported per model;
  settings are never pooled across models without stating so.
- Models: **Qwen/Qwen2.5-14B-Instruct** for both extractor and editor
  (identical model in both roles, `same_as_extractor: true`; reported as
  such). Source of truth: `config/experiment.json`.
  - `hf_model_id` (both roles): `Qwen/Qwen2.5-14B-Instruct`.
  - Serving: local `transformers` on the molab GPU session; no hosted API in
    the core run. Weights in bfloat16, no quantization.
  - Generation limits: extractor 256 new tokens; editor 1024 new tokens
    (`config/experiment.json`).
  - Model `revision` and `serving_library_version` are pinned and recorded
    from the loaded environment at first download, then carried into the cache
    key (`src/experiment_runner.py`); both are `null` in the config until that
  point. `os` is recorded as distro plus kernel string (`/etc/os-release`, `uname -a`). The same first-load record also holds GPU name, CUDA and driver version, torch and transformers versions, and the batch size B. Frozen Day-1 values: revision `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`, Transformers `5.20.2`, Torch `2.11.0+cu130`, OS `Linux-4.19.0-gvisor-x86_64-with-glibc2.41`, GPU `NVIDIA RTX PRO 6000 Blackwell Server Edition`, CUDA `13.0`, driver `595.71.05`, batch size `16`.
- **Scope: single-model pilot.** All claims are restricted to Qwen2.5-14B-Instruct and Test-H; there is no cross-model generality claim.
- **Stretch model (optional, pre-registered).** If the core run, the analysis tables and a report draft are complete with at least one day to spare, one instruct model from a different family may be run with the identical protocol and gates (oracle at least 85% on development, same screening, same limits). It is reported separately and never pooled with Qwen2.5-14B-Instruct. Its config block, revision pin and any added hash entries are logged as a Section 11 deviation dated when they are added.
- Compute: locally served open-weight models on the molab GPU session
  (NVIDIA RTX PRO 6000 Blackwell, ~96 GB; batched Transformers generation;
  measured 6.1 prompts/s in the Day-1 200-prompt benchmark). A hosted API is an optional
  control only and is not part of the core run.
- Decoding at temperature 0 (greedy) for the primary comparison. Greedy decoding is deterministic for a fixed batch size and composition on fixed hardware, but batched bfloat16 generation with padding can change a few outputs when batch size or composition changes. Controls: left padding, a fixed batch size B and fixed notice order, batch size and GPU recorded with every generation, and any out-of-memory batch halving logged as an event. The first cached response for a (model, prompt, notice) key is canonical; later regenerations never overwrite it. Every raw
  response is cached and all reported numbers come from the cache; cached
  replays are marked and excluded from timing summaries. GPU time is recorded
  per arm and reported as a cost (Section 7). Local serving also makes
  L-Probe affordable (four feedback evaluations over the existing section
  patches); it is reported as an exploratory localizer, outside the frozen
  set (Section 6).
- Prompt transport: the rendered task prompt is supplied as the user content
  of the pinned model tokenizer's chat template with
  `add_generation_prompt=True`. This wrapper is transport formatting and does
  not change the task prompt text or evaluator; a plain rendered-prompt
  fallback is used only for tokenizers without a chat-template API.

## 2. Data and splits

- 60 hand-written notices: 20 development, 40 final test (Test-H). Test notices and labels were read once for label-quality review before any model output existed; that review produced decisions A1 and A2 (Sections 11-12) and no result-based decision. After the freeze no test notice or label is inspected, and no decision uses test-set model output, until the single final evaluation. The evaluator code reads `data/test_labels.jsonl` only in that final pass.
- Development notices are split once, stratified by notice category, into
  **feedback set (10)** and **selection set (10)**, recorded in
  `data/split_manifest.json`.
  - Feedback set: the only source of the feedback packet.
  - Selection set: used only by the acceptance rule (Section 5).
  - Defect screening and oracle-prompt checks use all 20 development notices
    (no editor involved, so no leakage).
- Structural checks on the 60 notices (unique ids, schema, verbatim evidence,
  null notes, family isolation, split sizes, duplicate events, whitespace):
  `python scripts/validate_data.py`, nonzero exit on any violation.
- Each final prompt is evaluated on the test set exactly once.
- The optional generated test set (Test-G, ~300 templated notices) is
  **dropped from the core**; all claims are restricted to Test-H. Test-G is
  listed as an extension.

## 3. Defect screening and inclusion

- A defect is kept for a model only if it lowers strict record accuracy on
  the 20 development notices by at least **15 points** (3 notices) relative
  to the oracle prompt. This is a pragmatic threshold, not a statistical
  one. Dropped defects are reported with their accuracy drop.
- **Thin-category exception (pre-registered 25 October 2026):** when a
  defect's targeted development category contains fewer than 3 notices, the
  defect is kept if it lowers strict accuracy on the 20 development notices
  by at least **10 points** and **every notice of that category flips**.
  This applies to **D3 only** (2 explicit-correction notices). D3 is
  therefore a **thin-evidence setting**: all across-setting analyses are
  reported both with and without it.
- D2 can change at most the 4 deadline-distractor notices, so retention
  requires **3 of 4** to flip (maximum 20 points). Record this when reporting
  which defects survive screening.
- **No-feedback-signal rule (added in revision 2):** a setting is kept for the editing arms only if the defective prompt fails at least **1** feedback-set notice. Otherwise the editor has nothing to repair and L-Behav has nothing to rank; the setting is reported as "no feedback signal".
- The oracle prompt must reach at least **85%** strict accuracy on development for a model to be included; otherwise that model has no valid ceiling and is reported as excluded.
- **Oracle shakedown (pre-freeze, development notices only):** the oracle prompt is run once on the 20 development notices before the freeze. If it is below 85%, it may be revised using development notices only (never test notices, labels or outputs). Each revision is logged in Section 11 and `prompts/oracle.json` is re-hashed. This is the only model run before the freeze. The oracle's development accuracy is therefore optimistic; the test ceiling is the honest estimate.
- The original prompt is included only if it has at least 3 development
  failures (otherwise "no headroom", reported). Localization metrics are
  N/A for it; outcome metrics apply.
- If fewer than 3 settings survive screening for a model, all
  across-setting analyses for that model are descriptive only.

## 4. Arms

Defective (unchanged) | Rewrite | Span edit | Section patch + localizer X
(L-LLM, L-Rule, L-Behav; L-Probe exploratory) | Oracle-section patch |
Random-section patch | Oracle prompt | Restricted baseline (constraints and
output_format only).

- **L-Behav** (behavioral ablation): for each section, neutralize it by
  replacing that section with the matching section of `prompts/original.json`
  (frozen), rerun extraction on the feedback set, and rank sections by the
  share of failing feedback notices whose raw output changes. Uses
  feedback-set notices only.
- **Patch-all-sections design:** for every setting, a patch is generated for
  each of the four sections (same packet, same limits). Each localizer's
  result, the oracle-section result and the random-section result are then
  lookups of the corresponding section's patch. Random-section = mean over
  the four sections (the expectation), so no seed is needed. If no candidate
  is accepted after attempt 1 and the attempt budget remains, attempt 2
  regenerates the four section patches from the retained prompt with an
  updated packet; a `no_change` editor decision stops the branch.
- All editing arms share the same feedback packet, attempt limit (2), size
  limits (60 words of new text, 240-word prompt), acceptance rule (Section 5),
  and admissibility review. Rewrite and span edit are bound by the same
  240-word prompt limit.

## 5. Feedback packet, acceptance rule, admissibility

- **Packet:** for each failing feedback-set notice: notice text, model
  output, gold fields, and the failed field names. Cap of 5 failing notices
  plus 2 passing notices (to help preserve correct behavior), all chosen in
  fixed file order. Identical across arms.
- **Acceptance rule:** of the (up to) 2 candidate edits, accept the one with
  the highest strict accuracy on the selection set, but only if it is
  strictly higher than the current prompt's selection accuracy and introduces
  no new JSON/schema invalidity (invalid-output count must not increase).
  Ties between candidates go to the earlier attempt. If none qualifies, the
  prompt is unchanged (counts as a failed repair, not an exclusion).
- **Admissibility review:** a candidate is rejected if it (a) contains
  literal strings, names, dates or venues from any development or test
  notice, (b) exceeds size limits, (c) edits a section other than the
  targeted one in a section patch, or (d) changes the output schema. Checks
  (a)-(d) are automated first, then a manual pass. Rejections and reasons
  are logged and reported. Review is done blind to test results.

## 6. Localizers

Rankings over the four sections with fixed tie-break order: output_format,
constraints, instruction, context. Localizers use feedback-set data only
(or no data, for L-Rule). Selection and test data are never used by a
localizer; only the acceptance rule uses the selection set.

L-Probe is reported as an **exploratory** localizer: it scores each of the
four already-generated section patches on the feedback set and ranks sections
by the resulting strict accuracy. Local serving (Section 1) makes its four
feedback evaluations cheap. It is not part of the frozen localizer set used
for P1, does not change the tie-break order, and is labelled exploratory.

L-Rule maps observed error codes to sections (`src/localizers.py`); L-Behav
neutralizes each section by substituting the matching section of
`prompts/original.json` and ranks by the share of failing feedback notices
whose output changes.

## 7. Metrics (definitions)

- **Strict parse:** the raw response string is decoded, unmodified, as exactly
  one JSON object and nothing else. No Markdown-fence stripping, no trimming
  of surrounding prose, no regex extraction of an embedded object, no repair
  of the response. Duplicate keys and non-standard constants (`NaN`,
  `Infinity`) are rejected. A response that would parse only after cleaning
  is JSON-invalid.
- **JSON validity:** share of attempted responses that pass the strict parse.
- **Schema validity:** among strictly parsed responses, the value is a JSON
  object with exactly the keys `event`, `date`, `venue` (all present, no
  extras); `event` is a nonempty string; `date` and `venue` are each a
  nonempty string or JSON `null`. Schema-validity rate is reported over all
  attempted responses (schema-valid / attempted); JSON-valid but
  schema-invalid counts as invalid for strict accuracy.
- **Strict record accuracy:** response is JSON-valid, schema-valid, and all
  three fields equal gold after normalization. Primary outcome.
- **Normalization:** whitespace only - each run of whitespace becomes one
  space and ends are stripped, applied to both sides. No case folding, no
  punctuation or apostrophe normalization, no date reformatting, no
  Unicode/diacritic folding. A response differing from gold only in case or
  punctuation is incorrect.
- **Sentinel values:** gold `null` is satisfied only by JSON `null`. The
  strings `"null"` and `"N/A"` are non-null values (unwarranted filling where
  gold is null; a mismatch where gold is a string). `""` is schema-invalid.
  An omitted key is schema-invalid. None of these are interchangeable with
  `null` (`task_contract.md`, content rule 6).
- **Field accuracy:** per-field exact match after normalization, computed for
  every attempted response; a JSON- or schema-invalid response scores 0 on
  all fields and is never excluded from a denominator.
- **Lenient accuracy:** secondary diagnostic only (for example, after
  stripping fences or extracting an embedded object). It is never used for
  screening, acceptance, primary outcomes, or statistical tests, and is
  labelled as lenient wherever shown.
- Reference implementation: `src/evaluator.py` (strict parse, schema checks,
  whitespace-only comparison), pinned by `tests/test_evaluator.py`; dataset
  structure is checked by `python scripts/validate_data.py`.
- **Null-policy accuracy:** correct null where the notice lacks the field,
  non-null where present.
- **Unwarranted filling:** non-null value where gold is null.
- **Recovery rate:** among notices wrong under the defective prompt, the
  share correct under the edited prompt.
- **Regression rate:** among notices correct under the defective prompt,
  the share wrong under the edited prompt.
- **Restoration ratio:** (acc_edit - acc_defective) / (acc_oracle -
  acc_defective); reported unclipped; N/A when the denominator is 0.
- **Top-1 localization accuracy / MRR:** rank of the injected section;
  chance top-1 = 25%, reported next to the Random baseline.
- **Cost:** character edit distance, prompt growth (words), editor calls,
  extraction generations, tokens, GPU seconds per arm (local serving), and
  wall-clock time.
- **Strict localization** (localizer's top-1 equals injected section) and
  **fixability** (oracle-section patch restoration ratio > 0.8 on test) are
  reported separately. Localization accuracy is reported on all settings
  and on fixable settings only.

## 8. Statistical analysis

**Primary comparisons (confirmatory), outcome = strict accuracy on the test
set:**
- P1: Section patch with L-LLM (pre-specified primary localizer) vs Rewrite.
- P2: Oracle-section patch vs Random-section patch (does correct
  localization matter?).

Holm correction across P1 and P2. Everything else is exploratory and labelled so. **The confirmatory test for P1 and P2 is the across-setting analysis (item 2 below), with the setting as the unit.** Within-setting results (item 1) are per-setting estimates that support it; they are not separately corrected and no per-setting confirmation claim is made. (Clarified in revision 2: with up to 9 settings, per-setting tests would otherwise multiply the confirmatory comparisons.)

1. Within a setting (per-setting estimate; supports the across-setting test): **paired cluster bootstrap with the
   family as the resampling unit**. The 40 test notices form 14 families
   (2-4 near-duplicate notices each, `data/split_manifest.json`); notices in
   a family share template and vocabulary, so notices are not independent.
   For each comparison, correctness is paired per notice under both arms.
   Resample the 14 families with replacement 10,000 times, carrying all
   notices of each drawn family, and recompute the difference in strict
   accuracy (arm A - arm B) on the carried notices. Report the observed
   difference and its 95% percentile confidence interval (2.5th-97.5th
   percentiles). **The confidence interval carries the inference; p-values
   are descriptive only.** Seed 20261014; fixed resample count.
   Implementation: `src/stats.py` (`paired_cluster_bootstrap`), tested in
   `tests/test_stats.py`.
   Descriptively, also report (a) the bootstrap two-sided p-value
   (2 x min(share of resamples <= 0, share >= 0)) and (b) an **exact
   sign-flip test over the 14 families**: enumerate all 2^14 sign assignments
   on the family totals, statistic = signed sum of family totals / total
   notices, two-sided p = share of assignments with |statistic| >= |observed|.
   No seed is involved (exhaustive). Implementation:
   `scripts/sign_flip.py` (`exact_sign_flip`).
   An exact McNemar test over the 40 notices (notices treated as independent)
   is reported as a **labelled sensitivity analysis** for comparability with
   common practice, not as the primary inference.
2. Across settings (**confirmatory test of P1 and P2**): for each setting, the paired difference in strict test accuracy (arm A - arm B). Primary test: exact Wilcoxon signed-rank test with the setting as the unit (zero differences dropped; the number of nonzero differences is reported), Holm across P1 and P2. Companion: bootstrap over settings, 10,000 resamples, 95% percentile confidence interval of the mean difference. All settings come from one model and share the same 40 test notices, so they are not independent; claims are limited to "across these settings". Report with and without D3.
3. All arms reported, including losers and unchanged-prompt outcomes.
   Small setting counts mean tests are descriptive; avoid strong language.

**Pre-registered sensitivity analyses (report alongside, never as primary):**
- Exclude TEST-001 to TEST-004 (terminal-punctuation family; Section 12, A1), because no development notice exposes that pattern (13 test families remain). Report the oracle ceiling both raw and with these four excluded.
- Across-setting analyses both with and without D3, because D3 is a
  thin-evidence setting (Section 3).

**Power and limitations (report alongside the results):**
- The effective sample size for within-setting comparisons is **14
  families** (26 families across all 60 notices), not 40 independent notices.
  Cluster-bootstrap intervals will be wide; only large effects will have
  intervals excluding zero, and a non-significant result is not evidence of
  no effect.
- The development set contains exactly **one** explicit-correction notice and
  **one** unresolved-conflict notice in the feedback split, and one of each
  in the selection split. Any repair aimed at those behaviors therefore rests
  on a single example in the feedback packet and a single example in the
  acceptance set. This is a design limitation to state, not a tuning opportunity. Results for repairs aimed at those two behaviors (D3 and any conflict-targeting defect) are labelled **descriptive-only**: on them a repair can move selection accuracy by at most 10 points (one notice).
- **Single-model pilot:** all findings are restricted to Qwen2.5-14B-Instruct, one prompt format and Test-H. No cross-model or cross-task generality claim is made.
- The development-only oracle check cannot detect failure on the
  terminal-punctuation family (TEST-001 to TEST-004; Section 12, A1), so the
  oracle's test ceiling may be **near 90%** (4 of 40 notices).
- A model that never emits `null` can score at most **50% strict accuracy on
  the test set** (20 of 40 notices require at least one null: 12 venue nulls
  and 10 date nulls). Use this as baseline context when interpreting the
  null-policy defects (D1, D3, D4).

## 9. Seeds, determinism, exclusions

- Temperature 0 for the primary comparison; the core run is greedy (`sampled_repeats_S: 1`) and contains no sampled arm. If a sampled arm is run as an exploratory extension it uses 5 seeds (20261014 + i, i = 0..4) and reports the mean and range. Seed 20261014 also covers any tie-break randomness. Cache keys per `src/experiment_runner.py`.
- A generation or transport failure is retried up to 3 times
  (out-of-memory: halve the batch size and retry). If it still fails, the
  call is logged and the notice scored as incorrect; failures are counted
  and reported. No notice or setting is dropped after seeing test results.
- **Run storage and resumability (revision 2):** molab sessions are capped (12 h, idle shutdown) and a session's working directory may be rebuilt on restart. The response cache (append-only JSONL per arm) and the run manifest are synced to Google Drive after every arm (`scripts/sync_runs.sh`, rclone). The runner is resumable: it skips any (setting, arm, notice) already cached. Every session starts by writing a manifest (git commit, model revision, library versions, GPU, CUDA, driver, batch size); a change in any field between sessions is logged as a deviation. Credentials are never committed.

## 10. Frozen artifacts

SHA-256 integrity inventory recomputed on 28 October 2026, after environment
pinning, label review, and addition of the section editor template. This is a
current artifact-integrity record; it does not retroactively alter the frozen
analysis decisions. Reproduce with `bash scripts/hash_frozen.sh` and update
this block whenever a listed artifact changes.

```text
ca2003cd1e34ffd33d9313de2c0c7d056581b21b08d5cea702f972c756794622  task_contract.md
70afc778d6a721979c6c7d1a1a7b6b6b297f32a0d4bdcfcb01cd7ef596c37315  config/experiment.json
be4c6e2cd754b6ea9706d88c92885baa7fb204795c1250df414f401b6ff22431  data/feedback.jsonl
d7fd7207df56884a7d152795bd0759925cc7f470f8de9754d569dffd0fe595a3  data/selection.jsonl
39c91a5f2fd532193cecf69772c310a6996b561f32fa6be217e28e0e30a95e78  data/test_inputs.jsonl
6cff144787e9207ae7743f610793d77acf7e7ea7c0dc287489d7cf83e8a92248  data/test_labels.jsonl
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  data/generated_test_inputs.jsonl
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  data/generated_test_labels.jsonl
67a7cf343bd3d5c6aa9011df6a2a5c0add5dcfcb3602c3d712bd05722b5db213  data/split_manifest.json
71b582dd62dbe3cb18d473e7fc0e6d1357fbb4452219e52e9c77141dc464a4c5  prompts/defects/D1.json
c68cab35901dc13009356f0283fd43ef3686efdd2006b3bdbbcd2354e0ec0165  prompts/defects/D2.json
ed7213055c7d4fede913d36b8b994bd5d3b4be5b196e3292d22d3d984dfedee5  prompts/defects/D3.json
77652ddcd9ec5b336010516cde1755c979e25791c850079a583f8fc1385088ce  prompts/defects/D4.json
6961a6842417c6f978789487947bdc04e60cf3e34f049b07dbb7889a48b8d130  prompts/defects/D5.json
108edc4bc32861fbcd15cb2150009044bad949963e32bb0c4204515cb5ee969a  prompts/defects/D6.json
9020c9710845da260792cdc7e2b9066f03f202c3bd3f81c59e054da433212999  prompts/defects/D7.json
41225e4f2c55b91057533a6d055028acbd574cb0e8b6db9f124e2cc3458684da  prompts/defects/D8.json
304c489a3dc0615bc21957277888355fb68d9be7f015213696429b9c8d856a4e  prompts/oracle.json
ef9f7baf69cf9777c8495a22da07a94d2760e75ea1dd72056b41b468c1f784bf  prompts/original.json
54c28c80a2c9b4a4b9021c0f2ca768fa06a6b27dab834f4bbc9ebfdf9ea51583  prompts/rewrite_editor.txt
1164024646a84a6a8d8065837b45478fd77a0e52f793ce42630535dddbf39642  prompts/section_editor.txt
3afcacaa0f833eb3813cc8ff10f1b1195f5d5f0bfc761d3b6aba20727d21c2ce  prompts/targeted_editor.txt
```

`data/generated_test_inputs.jsonl` and `data/generated_test_labels.jsonl` are
empty placeholders for the dropped Test-G extension; they are hashed so the
full file set is covered. The hash script and this plan use
`data/split_manifest.json` (not `data/splits.json`).

The integrity inventory above was verified against the current frozen input files on 28 October 2026. The `prereg-v1` tag also fixes `src/`, `scripts/`, `tests/` and the dependency lockfile; record `git rev-parse prereg-v1` in every run manifest.

## 11. Deviations

- **D-01 — Qwen chat-template transport correction (27 October 2026,
  pre-freeze).** The first Molab shakedown passed the rendered prompt as plain
  text to the Qwen instruct checkpoint and produced 25% strict development
  accuracy, mostly malformed or repeated outputs. Before any optimization or
  final-test result, the adapter was corrected to wrap the same rendered task
  prompt as the tokenizer's user chat turn with `add_generation_prompt=True`.
  The corrected oracle shakedown reached 90% strict development accuracy
  (18/20), passing the 85% inclusion gate. The model, dataset, labels,
  evaluator, and acceptance rules were unchanged.

Each entry records: ID, date, what changed, why, and whether it was decided
before or after any test result was seen.

Pre-freeze amendments, all dated 25 October 2026 and made before any run and
before any test output existed; none are deviations. Any change after the
`prereg-v1` tag is recorded above.

- **A2 (contract amendment).** `task_contract.md` content rule 1 and
  `prompts/oracle.json` constraints now state that the venue is the full
  location phrase as stated. Noticed during the second-pass test-label review
  (TEST-002, `Room 104 of the Science Block`); gold labels, notices and data
  files are unchanged. Both amended files are re-hashed in Section 10.
- Metric definitions, strict parse and sentinel rules (Section 7).
- Family-cluster bootstrap, exact sign-flip test and limitations
  (Section 8).
- D3 thin-category screening exception and the D2 3-of-4 note (Section 3).
- Local serving record, GPU-time cost, L-Probe exploratory wording, and a narrowed determinism note (fixed-batch determinism; Sections 1, 6, 7, 9).
- Config: bfloat16, molab Linux, extractor 256 / editor 1024 tokens
  (Section 1).
- **Revision 2 (25 October 2026, before any run; none are deviations):**
  - Section 2 wording corrected to record that test notices were read once for label review.
  - Confirmatory unit for P1 and P2 is the across-setting analysis (Section 8).
  - No-feedback-signal rule and oracle shakedown (Section 3).
  - Seeds policy reworded; `sampled_repeats_S: 1` (Section 9).
  - Determinism note narrowed; batch size and GPU recorded; first cached response canonical (Sections 1, 9).
  - Single-model pilot scope; stretch model pre-registered (Section 1).
  - Correction and conflict repairs labelled descriptive-only (Section 8).
  - Run storage and resumability on Google Drive (Section 9).
  - Oracle ceiling reported raw and excluding TEST-001 to TEST-004 (Section 8).

## 12. Resolved pre-freeze items (25 October 2026)

All resolved before any run; no test output existed. The binding rules live in
Sections 1-8; this section records the decisions and the evidence behind them.

- **A1 (terminal punctuation, TEST-001 to TEST-004): Option 2.** Gold and
  notices unchanged. Every arm sees the same four notices, so this is
  ceiling/baseline noise rather than arm bias. A pre-registered sensitivity
  analysis excludes TEST-001 to TEST-004 (Section 8). The development-only
  oracle check cannot detect failure on this family, so the oracle's test
  ceiling may be near 90% (4 of 40 notices).
- **A2 (multi-part venue, TEST-002): Option 1.** `task_contract.md` content
  rule 1 and `prompts/oracle.json` constraints now state that the venue is the
  full location phrase as stated, not a truncated part of it. Noticed during
  the second-pass test-label review; logged as a dated pre-freeze amendment
  (Section 11). Gold labels, notices and data files unchanged; both amended
  files re-hashed (Section 10).
- **D (screening power): thin-category exception adopted.** For a defect whose
  targeted development category has fewer than 3 notices, retention requires
  a 10-point drop on all 20 development notices and every notice of that
  category flipping. This applies to D3 only (2 explicit-correction notices),
  which is flagged as a **thin-evidence setting**; every across-setting
  analysis is reported with and without it. D2 can affect at most the 4
  deadline-distractor notices, so it requires **3 of 4** to flip (Section 3).
- **Inference:** the family-cluster bootstrap confidence interval carries the
  inference; p-values are descriptive. The exact sign-flip test over the 14
  families is included as a descriptive companion (Section 8,
  `scripts/sign_flip.py`).
- **Serving:** local `transformers` on the molab GPU. GPU time is a reported
  cost, L-Probe is described as an affordable exploratory localizer, and the batched-nondeterminism caveat is narrowed to fixed-batch determinism (revision 2; Sections 1, 9).
