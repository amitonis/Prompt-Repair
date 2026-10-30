# Research-informed improvement plan

Updated: 30 October 2026

## What the current evidence says

- The frozen primary test has 40 notices in 14 families. The leading prompt arms share five failures: three event-title boundary/punctuation cases and two unresolved-date cases. The paired targeted-repair-versus-rewrite comparison had no discordant notices. This does not show that one prompt method is better.
- Exploratory v2 was 11/12 for the oracle baseline and 10/12 for the boundary/conflict candidate. The candidate's extra error included an article in the event name (`The Film Poster Workshop`). The shared EB-005 error depends on an annotation convention the contract does not define.
- Exploratory v3 was 12/12 for both arms, including valid JSON and schema on every case. The tied ceiling result cannot distinguish the prompts. V3 is a targeted challenge set drafted after observing v2 and the primary errors; treat it as post-hoc diagnostic evidence, not a blind independent test.
- The exploratory results do not establish that the v2 candidate would improve the 40-case primary test. They must not be combined with primary scores.

## Highest-value improvements

### 1. Resolve the annotation boundary before collecting more labels

Write explicit, positive and negative examples for event-title boundaries: terminal punctuation, punctuation internal to a title, quotation marks, articles such as “The”, organizer names, appositives, and title/organizer word order. Decide whether punctuation is part of the title from the stated title semantics, not from a hidden convention. Independently annotate a small pilot and adjudicate disagreements before freezing labels. Preserve the frozen primary labels and scores; record any newly discovered ambiguity as an audit finding.

This directly addresses the current strict-score bottleneck. Annotation research distinguishes genuine ambiguity from annotator error and recommends evaluating agreement where it is expected, rather than forcing every disagreement into a single unexplained convention ([Sommerauer et al., COLING 2020](https://aclanthology.org/2020.coling-main.422.pdf); [Wong & Lee, 2013](https://aclanthology.org/W13-4307.pdf)).

### 2. Build a genuinely held-out, more discriminating evaluation

Use separate data for prompt development, candidate selection, and final evaluation. The next set should be authored or sampled without access to candidate outputs, span multiple independent event families, and include realistic wording rather than only explicit cues. Stratify the hard cases: organizer/title ambiguity, punctuation, corrections, unresolved conflicts with varied source cues, missing fields, distractor dates, and full location phrases. Ensure enough cases per stratum to report counts meaningfully; one case per category, as in v3, is only a smoke test.

The v3 draft was new but purpose-built after earlier errors were known. Keep its 12/12 result as a targeted challenge-set result. Do not use it to claim generalization or to tune the candidate and then reuse it as a final test. Work on true few-shot selection shows that prompt choice is difficult without held-out data and that data-driven selection can be unreliable with very small samples ([Perez et al., 2021](https://arxiv.org/abs/2105.11447)).

### 3. Set a minimum detectable improvement, then calculate sample size

Before commissioning another set, specify the smallest improvement that would matter (for example, a chosen number of exact-record cases per 100) and the comparison to test. Estimate uncertainty from paired per-case differences, clustering resamples by family. Calculate the required number of independent families from the observed paired variance and target power; do not select an arbitrary sample size. Miller's framework recommends paired comparisons, reporting uncertainty on the difference, and planning size against a minimum detectable effect. Its worked example reaches roughly 969 questions for a 3-point effect under its particular assumptions, illustrating why a 40-item test cannot resolve small changes; that number is not a universal target ([Miller, 2024](https://arxiv.org/html/2411.00640)).

The existing analysis already uses family-level resampling for the primary comparisons. Keep that approach and add a prospective power/MDE calculation before a new confirmatory study.

### 4. Keep the metric aligned with the contract

Keep exact JSON, schema, and field-value checks: the task contract requires verbatim extraction and exact keys, so a semantic judge could incorrectly accept paraphrases, altered dates, or a wrong null. Improve diagnosis by reporting event/date/venue field accuracy and error types alongside exact-record accuracy. Audit ambiguous gold spans separately; do not silently relax exact match or change frozen labels.

Prompt-sensitivity research finds that model rankings can shift under instruction paraphrases, and a 2025 study argues some measured sensitivity comes from rigid answer matching on open-ended outputs ([Sun et al., 2023](https://arxiv.org/html/2306.11270v2); [Hua et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1006/)). The second result is a reason to align evaluation with the task definition, not to replace this project's verbatim metric with an LLM judge.

### 5. Treat automatic prompt optimization as a later comparison

Methods such as ProTeGi use natural-language critiques of minibatch errors to propose prompt edits; MIPRO searches over data-grounded instructions and demonstrations; GEPA reflects over execution traces and evolves prompt candidates ([ProTeGi, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.494/); [MIPRO, EMNLP 2024](https://arxiv.org/abs/2406.11695); [GEPA, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0e9e708b6f48e14fd0ac29e167413f76-Abstract-Conference.html)). These are promising optimization methods, but their reported gains depend on task, data, and compute. With the current small reviewed sets, running an optimizer would mostly search for quirks of those cases. First build a larger reviewed development set and keep a sealed, adequately powered test set. Then compare one optimizer to the existing manual repair process under the same inference budget.

## Recommended order of work

1. Freeze the completed primary and exploratory results as reported.
2. **Completed:** added field-level and error-subtype diagnostics from saved outputs; no prompt or label changes. See `runs/FIELD_DIAGNOSTICS.md` and `runs/field_diagnostics.json`.
3. Have an independent reviewer audit the future title/punctuation annotation guide and the old ambiguous examples; document, but do not rescore, the frozen study.
4. Define a minimum detectable effect and power calculation for a new study.
5. Create a larger, independently authored and double-reviewed development set plus a sealed test set; use family-level grouping and realistic hard cases.
6. Compare the frozen oracle, the v2 candidate, and one constrained new method (for example, a trace-based repair or a lightweight deterministic validation layer). Keep prompts and compute budgets fixed before opening the sealed labels.

## Bottom line for this project

The immediate improvement is better evidence, not another prompt edit. Current outputs are already schema-valid in the reviewed sets; the unresolved questions are exact title boundaries and whether a prompt change generalizes. A fresh, sufficiently large, independently labeled evaluation is needed before claiming improved repair quality.
