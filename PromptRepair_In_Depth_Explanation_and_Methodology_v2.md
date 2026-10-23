# PromptRepair
## Targeted Prompt Correction for Structured Text Extraction

**In-depth explanation, methodology, implementation blueprint, and two-week delivery plan**

| Item | Project specification |
|---|---|
| Subject | Prompt Engineering |
| Original project | Research Project 57 / PE-13: PromptRepair: Automatic Repair Instead of Complete Prompt Rewriting |
| Source location | Project compendium, page 65 |
| Demonstration domain | Extracting information from fictional college event notices |
| Timeline | 14 days |
| Compute | One RTX 6000-class GPU on marimo molab (stated by the student; verify with `nvidia-smi`); local open-weight models; no paid API required |
| Inference assumption | Open-weight instruct models served locally inside the molab session; a hosted API is an optional control only |
| Planned stack | Python, marimo notebooks on molab, Hugging Face Transformers or a serving library such as vLLM, local JSON/CSV files, GitHub and Google Drive for persistence |
| Study size | 60 hand-written labelled notices (20 development, 40 final test) plus an optional generated test set of about 300 notices |
| Main comparison | Starting prompt vs complete rewrite vs span-level edit vs one-section patch (with several fault localizers), plus oracle-prompt, oracle-section and random-section controls |
| Document status | Proposed methodology (revised); the application and all LLM experiments have not been run |
| Prepared | 14 October 2026; revised 19 October 2026 (GPU, related work, novelty) |

> **Revision of 19 October 2026.** This version changes three things relative to the 14 October draft:
>
> 1. **Compute.** The student now has a GPU on marimo molab (stated as an RTX 6000-class card). Extraction and editing therefore run on locally served open-weight models rather than a hosted API, and the request-quota planning of the earlier draft is replaced by a throughput and session-time plan (Section 14).
> 2. **Related work.** Published prompt-optimization work is mapped to this project, with citations (Section 1A and Section 20).
> 3. **Novelty.** The earlier draft was a replication-scale comparison. Section 1B states what is and is not new, and Section 9A adds the experiments that carry the novelty claims.
>
> **Honest status of novelty:** closely related published work exists, most directly Local Prompt Optimization [L5]. The claims in Section 1B are deliberately narrow, rest on a limited literature search done on 19 October 2026, and must be re-checked with a full search before being asserted in a paper.

> **The central distinction:** this project repairs a reusable **prompt**, not merely one incorrect model response. The repaired prompt must be tested on other notices that were not used to create the repair.

## Contents

1. [Source basis and the two-week adaptation](#1-source-basis-and-the-two-week-adaptation)
   - 1A. [Related work and positioning](#1a-related-work-and-positioning)
   - 1B. [Novelty and contributions](#1b-novelty-and-contributions)
2. [Project explanation and motivation](#2-project-explanation-and-motivation)
3. [Research question, objectives, and boundaries](#3-research-question-objectives-and-boundaries)
4. [Theory and laboratory alignment](#4-theory-and-laboratory-alignment)
5. [Exact task definition and output contract](#5-exact-task-definition-and-output-contract)
6. [Dataset construction and separation](#6-dataset-construction-and-separation)
7. [Architecture and module responsibilities](#7-architecture-and-module-responsibilities)
8. [Prompt design and complete templates](#8-prompt-design-and-complete-templates)
9. [End-to-end experimental methodology](#9-end-to-end-experimental-methodology)
   - 9A. [Novelty experiments: defect injection and fault localization](#9a-novelty-experiments-defect-injection-and-fault-localization)
10. [Worked examples and failure analysis](#10-worked-examples-and-failure-analysis)
11. [Metrics and interpretation](#11-metrics-and-interpretation)
12. [Fairness, reproducibility, and limitations](#12-fairness-reproducibility-and-limitations)
13. [Application design and project structure](#13-application-design-and-project-structure)
14. [GPU compute on molab, throughput budget, and resilience](#14-gpu-compute-on-molab-throughput-budget-and-resilience)
15. [Fourteen-day implementation plan](#15-fourteen-day-implementation-plan)
16. [Testing and acceptance criteria](#16-testing-and-acceptance-criteria)
17. [Presentation, report, and viva preparation](#17-presentation-report-and-viva-preparation)
18. [Reference evaluator](#18-reference-evaluator)
19. [Final deliverables and extensions](#19-final-deliverables-and-extensions)
20. [Sources](#20-sources)

---

## 1. Source basis and the two-week adaptation

### 1.1 What the supplied project PDF actually proposes

The project entry asks whether modifying a faulty component is more effective than rewriting the entire prompt. Its method is to identify a failing component using development examples, edit that span, and compare the result against complete regeneration. The full proposal calls for **300 prompts across four tasks**, held-out evaluation, and measurements including accuracy improvement, edit distance, token growth, generalization, and iterations to success. It also suggests restricting edits to individual components, such as instructions, examples, constraints, or formatting. [P1, p. 65]

That is a broader study than the version described here. The PDF does **not** specify a college-notice dataset, a Streamlit interface, a 60-example study, or a 14-day schedule.

### 1.2 What this guide deliberately changes

The following are **proposed engineering decisions for your constraints**, not requirements copied from the PDF.

| Dimension | Original proposal | Proposed classroom prototype |
|---|---|---|
| Scope | Four tasks | One structured extraction task |
| Study objects | 300 prompts | A small family of starting prompts (one vague original plus injected-defect variants of an oracle prompt), their candidate revisions, and the notice sets |
| Editable components | Several component-level experiments | One section per attempt: constraints or output format in the restricted baseline; any of the four sections when a localizer chooses (Section 9A) |
| Baselines | Multiple optimization and refinement methods | Original, complete rewrite, span-level edit, targeted section repair, plus oracle and random controls |
| Iteration | Not capped in the entry | At most two editing attempts per optimization method |
| Deliverable | Minimal-edit prompt optimizer | Working local application plus a small controlled evaluation |
| Generalization claim | Across a larger task collection | Only to unseen notices within this chosen domain |

**Sixty notices are not sixty independent prompt-optimization problems.** This prototype investigates one task and a small family of starting prompts (Section 9A). It is a pilot implementation of Project 57, not completion of the full research specification.

### 1.3 How to read this document

Statements labelled **[P1]**, **[P2]**, or **[P3]** come from the supplied PDFs. Statements labelled **[W1]** through **[W6]** concern implementation documentation, **[M1]** through **[M4]** concern the marimo molab platform pages, and **[L1]** through **[L18]** are published literature found by web search on 19 October 2026 (abstracts and retrieved passages only). Architecture, dataset sizes, templates, algorithms, and schedules without those labels are the proposed design.

All model outputs shown as examples are **illustrative**, not observations from an executed LLM experiment. The blank results table later in this guide must remain blank until the actual evaluation is run.

---

## 1A. Related work and positioning

Everything in this section was found by web search on 19 October 2026 and is a **starting map, not a systematic review**. Citation keys [L1]-[L18] refer to Section 20. Where a statement says what a paper "reports", it paraphrases the abstract or a retrieved passage; read the paper before relying on any claim or number.

### 1A.1 Prompt optimization with LLM-proposed edits

| Work | What it does | Relation to PromptRepair |
|---|---|---|
| APE [L1] | An LLM proposes candidate instructions and the best one is selected by score | Global generate-and-select; a rewrite-style baseline family |
| OPRO [L2] | An LLM proposes new prompts conditioned on earlier prompt-score pairs | Score-guided global proposals; no component restriction |
| ProTeGi [L3] | Builds natural-language "gradients" from minibatch errors, edits the prompt against them, and searches with beam search and bandit selection | Closest ancestor of error-driven editing; edits are free-form over the whole prompt |
| PE2 [L4] | Studies how to design the meta-prompt (detailed descriptions, context specification, a step-by-step reasoning template) and reports targeted edits | Source of editor-template ideas (Section 8.4) |
| PRewrite [L16] | Trains a prompt rewriter with reinforcement learning | A learned complete-rewrite approach; not run here |
| TextGrad [L8] | Propagates LLM-written textual feedback to improve components of a compound system | "Components" means parts of a pipeline, not sections of one prompt |
| DSPy / MIPRO [L6, L7] | Optimizes instructions and demonstrations of LM programs against a metric | Broader pipeline optimizers; stronger but heavier baselines |
| GEPA [L9] | Reflective prompt evolution with Pareto-based candidate selection; reports gains over GRPO and MIPROv2; open-source code | Strongest external baseline if time allows |

### 1A.2 Work closest to the PromptRepair question

| Work | Why it matters |
|---|---|
| **Local Prompt Optimization (LPO)** [L5] | Asks the proposal LLM to mark, with edit tags, which tokens of the prompt should change, then edits only there. It plugs into APE, APO and PE2 and reports improvement over global optimization. This is the nearest prior work to "local repair versus complete rewrite". |
| **HAPO** [L11] | Attributes errors and edits functional prompt segments ("semantic units"), motivated by prompt drift. Shows segment-level editing already exists. |
| **SkillOpt** [L14] | Treats the number of edits per step as a learning-rate analogue, with patch and rewrite modes, for agent skill documents. Shows bounded edits versus rewriting has been studied in a neighbouring setting. |
| **GradPO** [L12], **GMPO** [L13] | Use gradients of a white-box model's loss to choose which prompt spans or segments to edit. Precedent for a gradient-based localizer. |

### 1A.3 Prompt drift and regressions

StraGo [L10] and HAPO [L11] describe prompt drift: an edit fixes some failures while breaking cases that previously worked. StraGo analyzes this by comparing how many errors an optimized prompt fixes with how many previously correct cases it breaks, which is the same idea as this project's recovery and regression rates (Section 11.2). Contrastive reflection [L15] adds nearby successful examples to the feedback and reports that failure-only feedback breaks more previously correct cases; this project's feedback packet includes passing examples for the same reason. **Cite these when reporting recovery and regression; do not present the idea as new.**

### 1A.4 Why locating the fault is hard

An interview study of developers who build software containing prompts [L17] reports that finding the cause of a prompt defect is hard, that participants often fell back on trial-and-error changes, and that changes sometimes introduced new errors. A taxonomy of prompt defects [L18] organizes defects into categories such as specification defects and structure/format defects; Section 9A uses it as the basis of the defect catalog.

### 1A.5 What the literature does not directly answer (from this limited search)

1. Most evaluations use real tasks where the true faulty part of the prompt is unknown, so *localization* quality is judged only indirectly, through final accuracy.
2. LPO's localization is LLM-judged [L5]; gradient-based localizers [L12, L13] are judged by downstream results. No retrieved paper compares several localizers against ground-truth fault locations.
3. I did not find a benchmark that injects known defects into a known-good, sectioned prompt. This is a statement about my search, not a proof of absence.

---

## 1B. Novelty and contributions

### 1B.1 Is the original design novel?

Not by itself. The original PromptRepair pilot (original prompt vs complete rewrite vs targeted repair on one extraction task) is a small, well-controlled comparison of an idea that LPO [L5], HAPO [L11] and SkillOpt [L14] already explore in other settings. It is a good course project and a sound replication-scale study, but it should not be described as a new method.

### 1B.2 Proposed contributions

Each contribution is a **claim to be tested**, listed with its closest prior work and its main risk.

| ID | Contribution | Closest prior work | Main risk |
|---|---|---|---|
| C1 | **Ground-truth prompt-defect benchmark.** Start from a known-good sectioned prompt, inject one known defect into one section, and keep that section as the ground-truth fault location. Localization and repair can then be scored directly. | Defect taxonomy [L18]; evaluations on real tasks with unknown fault location [L3, L5, L11] | Injected defects may look artificial; mitigated by also keeping the naturalistic vague prompt of Section 8.2 |
| C2 | **Controlled comparison of section-level fault localizers** (LLM-judged, error-code rules, ablation importance, probe-and-select, and in Phase 2 gradient attribution) against ground truth, with random and oracle bounds. | LPO's LLM-judged localization [L5]; gradient-guided selection [L12, L13] | Localizers may tie; a null result is still a result |
| C3 | **Edit-granularity ladder under matched budgets:** complete rewrite vs span-level edit (LPO-style) vs one-section patch, reported with restoration ratio, recovery, regression, and edit size. | LPO [L5], SkillOpt [L14], StraGo's drift analysis [L10] | The effect may depend heavily on the model and editor |
| C4 (Phase 2) | **Repair portability:** a repair made with model A is applied to models B and C, to see whether small edits transfer differently from rewrites. | Transfer across model families is raised in the GMPO abstract [L13]; no direct patch-vs-rewrite portability study was located | Needs several models and more time |

### 1B.3 Why the GPU makes these feasible

C1 and C2 need many optimization runs: every (model, defect, arm) combination is a separate run, localizers cost several extra candidate evaluations, and statistics need repeated measurements. Under the earlier API-quota plan this was impractical. A local GPU also exposes token log-probabilities and gradients, which hosted APIs generally do not, enabling the Phase 2 gradient localizer.

### 1B.4 How strong is this?

- C1 + C2 on one task and one or two models is plausibly a **workshop-scale** contribution if the results are clean and the claims stay narrow.
- A **main-track-scale** paper would typically need more tasks (the original proposal names four [P1]), stronger external baselines (an LPO-style optimizer, GEPA [L9], ProTeGi [L3]), more models, and a larger defect catalog. That is a multi-month plan, not the 14-day core.
- Do not claim: that targeted repair "beats" rewriting in general; that the benchmark is exhaustive; that gradient attribution works (it is a hypothesis); or that recovery/regression metrics are new (Section 1A.3).

### 1B.5 Suggested claim wording (adapt after results)

> "We introduce a defect-injection protocol for prompts in which the faulty section is known by construction, and use it to compare section-level fault localizers and edit granularities on a structured-extraction task."

Change "introduce" and "compare" to match what was actually run. Re-run the literature search (Google Scholar, Semantic Scholar, ACL Anthology, arXiv) before writing this sentence in a paper.

---

## 2. Project explanation and motivation

### 2.1 The problem in everyday language

Imagine asking an AI system to extract an event name, date, and venue from a college notice. The response may look useful but still be unsuitable for a program: it could include an introductory sentence, invent a missing venue, confuse a registration deadline with the event date, or omit a required key.

A common reaction is to replace the entire prompt with a longer one. This project tests a more restrained alternative: **find the observed failure, identify a plausible prompt component to change, make a small correction, and measure what happens**.

For example, an initial prompt may contain:

```text
Constraints: Keep the result concise and use the notice.
```

After observing unsupported values on development examples, a proposed repair could be:

```text
Constraints: Use only information explicitly stated in the notice.
Return null for an absent or unresolved date or venue. Do not guess.
```

The instruction, context, and output-format section remain unchanged in that repair attempt. This produces an inspectable change rather than an unexplained replacement of the entire prompt.

### 2.2 What a user would do

The application accepts a structured prompt and a small labelled development dataset. It runs the prompt, displays failures, proposes a constrained edit, highlights the change, and evaluates the candidate. It also offers a complete-rewrite baseline so the targeted approach has a meaningful comparator.

After optimization, the application evaluates the frozen prompt versions on the reserved test notices. A dashboard shows whether improvements came from correct content, better formatting, or both.

### 2.3 What makes this more than a wrapper

The useful contribution is the experimental and validation machinery: a fixed task contract, labelled examples, deterministic checks, controlled edits, comparable baselines, separate test data, regression tracking, and saved run records.

An interface that only says "Here is an improved prompt" does not establish improvement. The project becomes convincing when it can also say: "This section changed, these development failures motivated it, and this is its measured performance on previously unused notices."

### 2.4 What the system cannot know automatically

An observed output error does not prove which words in the prompt caused it. The diagnosis stage proposes a **likely repair location**. Testing then provides evidence about whether that intervention helped in this setting.

Similarly, an LLM's explanation of its edit is not an independent evaluation. Final correctness comes from the predefined labels and checks, not from the editor claiming that its own prompt is better.

---

## 3. Research question, objectives, and boundaries

### 3.1 Main research question

> Under the same model, development data, and editing-attempt limit, how does targeted prompt repair compare with complete prompt rewriting for extracting structured information from college event notices?

### 3.1A Additional research questions (novelty)

- **RQ2 (localization).** When a prompt contains one injected defect in a known section, which localization method most often identifies the correct section, and does better localization lead to better repair?
- **RQ3 (granularity).** Under matched attempt limits, how do a complete rewrite, a span-level edit, and a one-section patch compare in restoring a defective prompt's accuracy, and how many previously correct cases does each break?
- **RQ4 (portability, Phase 2).** Do repairs made with one model carry over to other models?

The main research question above is RQ1. RQ2-RQ4 are tested in Section 9A.

### 3.2 Working hypothesis

Targeted repair may improve compliance while changing less prompt text than complete rewriting. It may also preserve some previously correct behavior. These are hypotheses: the rewrite may perform better, both methods may tie, or both may fail to improve the original.

A useful study reports whichever outcome occurs.

### 3.3 Objectives

The core objectives are to build a component-based prompt representation; identify observable extraction and formatting failures; generate and enforce local edits; compare three frozen prompt versions; and document correctness, regressions, editing effort, and inference usage.

The prototype should demonstrate an entire prompt-engineering cycle rather than maximize application features.

### 3.4 Scope boundary

| Included in the 14-day version | Excluded from the 14-day version |
|---|---|
| English, text-only, fictional academic notices | Medical or cybersecurity applications |
| One clearly identifiable target event per notice | Arbitrary multi-event document understanding |
| Three output fields | Large or deeply nested schemas |
| Open-weight model(s) served locally for extraction and editing | Model training or fine-tuning (the GPU is used for inference only) |
| Local rule-based validation | An LLM judge as the final correctness authority |
| Several editing methods (rewrite, span edit, section patch), capped attempts | Open-ended autonomous optimization |
| Local web interface and saved result files | Accounts, payments, cloud deployment, or database administration |
| Plain text input | PDF upload processing, OCR, images, or document retrieval |

The PDFs are **design references**, not runtime documents the application must ingest.

### 3.5 Resource assumption

The student has access to a GPU on marimo molab, stated as an RTX 6000-class card. The molab pages reviewed for this revision [M1, M2] confirm GPU-backed notebooks on CoreWeave infrastructure, sessions of up to 12 hours, and a free public preview, but they do **not** name the GPU model or its memory. Run `nvidia-smi` on Day 1 and record the exact card and VRAM. The plan is written to work on 48 GB or more and to degrade gracefully, with quantization, on smaller cards.

What the GPU changes: extraction and editing run on locally served open-weight models; there is no request quota to budget; repeated runs, several models and many defect variants become practical; and white-box signals (log-probabilities, gradients) become available.

What it does not change: this is still **inference-only** use (no fine-tuning); results describe the chosen open-weight models, not hosted frontier models; and the free preview carries no availability guarantee. Keep a fallback (Section 14.9).

---

## 4. Theory and laboratory alignment

### 4.1 Theory mapping

The theory syllabus lists prompt anatomy in Unit 2; prompt chaining, iterative refinement, reusable templates, and incomplete inputs in Unit 3; evaluation and A/B testing in Unit 4; and simple LLM applications and an end-to-end capstone in Unit 5. [P2, p. 2]

| Syllabus topic | Proposed project evidence |
|---|---|
| Unit 2: instruction, context, input, output format, constraints | Explicit prompt sections and an independently defined output contract |
| Unit 3: iterative prompt refinement | Evaluate, edit, retest, and accept or reject a candidate |
| Unit 3: reusable templates | Use the selected prompt unchanged across new notices |
| Unit 3: ambiguous or incomplete inputs | Missing information and unresolved conflicts mapped to null |
| Unit 4: accuracy and A/B testing | Compare the same test notices under all three prompt versions |
| Unit 4: consistency and hallucination reduction | Report formatting failures, regressions, and unwarranted field filling |
| Unit 5: automation and simple applications | Local UI, automated evaluation, saved artifacts, and report export |

This is an implementation mapping, not a claim that the project covers every topic in every unit. For example, it does not include translation, code generation, or a complete comparison of zero-shot, one-shot, and few-shot methods.

### 4.2 Laboratory mapping

The strongest match is **Experiment 5**, which asks students to inspect an initial response, identify limitations, add constraints or structure, and compare the refined response. **Experiment 6** concerns controlled formats and JSON validation. **Experiment 7** concerns reusable templates. [P3, pp. 6-8]

The sequence of extraction, analysis, and a subsequent action also applies the chaining concept in **Experiment 4**, although its example uses student feedback rather than event notices. [P3, pp. 5-6]

Experiments 1, 8, and 10 provide additional conceptual connections through structured prompting, automated pipelines, and evaluation of academic tools. These are supporting links, not a promise that this single project reproduces each complete lab exercise. [P3, pp. 2-3, 9-11]

---

## 5. Exact task definition and output contract

### 5.1 Why define a contract first?

A prompt cannot be judged as correct without an agreed target. The task contract is the written specification used to create labels and validate responses. It must be fixed before optimization.

The contract is available to both editing methods. The extractor receives the current prompt version and the notice, not the notice's reference answer. The original prompt may incompletely express the intended contract; repairing that omission is part of the task. Do not claim the experiment changes wording only while preserving every semantic detail of the initial, incomplete prompt.

### 5.2 Contract for this prototype

Every input contains one clearly named target college event. Return exactly one JSON object with these fields:

| Field | Allowed value | Meaning |
|---|---|---|
| `event` | Nonempty string | The stated name of the target event |
| `date` | Nonempty string or JSON `null` | Its explicitly stated event date |
| `venue` | Nonempty string or JSON `null` | Its explicitly stated venue |

Copy the relevant wording rather than reformatting dates. For example, preserve `20 October 2026` instead of converting it to `2026-10-20`. Evaluation may collapse repeated whitespace, but it will not treat arbitrary paraphrases or date formats as equivalent.

An absent date or venue becomes `null`. So does a field with two unresolved conflicting values. An explicit correction such as "rescheduled from X to Y" selects Y; it is not an unresolved conflict. A registration deadline, publication date, or previous event date is not the current event date.

Every key must be present. `null`, `"null"`, `"N/A"`, an empty string, and an omitted key are not interchangeable. Do not include additional keys, Markdown fences, explanations, or text outside the JSON object.

### 5.3 Local validation schema

This schema is a local checking specification. In the main experiment, it is **not** supplied as a provider-enforced response schema, because doing so could obscure the effect of output-format prompting.

```json
{
  "type": "object",
  "properties": {
    "event": {"type": "string", "minLength": 1},
    "date": {
      "anyOf": [
        {"type": "string", "minLength": 1},
        {"type": "null"}
      ]
    },
    "venue": {
      "anyOf": [
        {"type": "string", "minLength": 1},
        {"type": "null"}
      ]
    }
  },
  "required": ["event", "date", "venue"],
  "additionalProperties": false
}
```

Add an application check rejecting whitespace-only strings. Also reject duplicate JSON keys and nonstandard numeric constants rather than silently accepting them.

### 5.4 Example

**Fictional source notice:**

```text
The AI Club announces the Prompt Engineering Workshop.
The workshop will take place on 20 October 2026.
The venue will be announced later.
```

**Reference answer:**

```json
{
  "event": "Prompt Engineering Workshop",
  "date": "20 October 2026",
  "venue": null
}
```

"The venue will be announced later" does not identify a venue. It should not become the venue string.

---

## 6. Dataset construction and separation

### 6.1 Proposed size and composition

Prepare 60 short fictional notices, typically around 40-100 words. Human-check every reference answer. The following distribution is a proposed blueprint, not a dataset already supplied by the PDFs.

| Primary case category | Feedback subset | Selection subset | Final test | Total |
|---|---:|---:|---:|---:|
| Complete, unambiguous information | 2 | 2 | 8 | 12 |
| Venue absent or not yet announced | 2 | 2 | 8 | 12 |
| Event date absent | 2 | 1 | 6 | 9 |
| Multiple dates with a deadline distractor | 2 | 1 | 6 | 9 |
| Explicit date or venue correction | 2 | 1 | 6 | 9 |
| Unresolved date or venue conflict | 2 | 1 | 6 | 9 |
| **Total** | **12** | **8** | **40** | **60** |

The 12 feedback examples plus 8 selection examples form the previously agreed **20-example development set**. The subdivision improves discipline without increasing the dataset size.

### 6.2 Purpose of each subset

**Feedback subset: 12 notices.** The editor may see these notices, their reference answers, the extractor's responses, and error reports. These examples support repair proposals.

**Selection subset: 8 notices.** The program evaluates candidates here and uses the scores to decide which version to retain. Do not place these notices or labels in the editor's request. Because selection scores influence decisions, this is still development data, not an independent final test.

**Final test: 40 notices.** These are used only after candidate selection is over. The extractor receives the notice text, while the evaluator separately holds its label. No final test error may be used to modify a prompt while continuing to report the same test set as unseen.

### 6.3 Prevent near-duplicate leakage

Changing only an event name and date does not create a genuinely different notice style. Give structurally related examples a `family_id` and keep the entire family within a single subset.

The broad categories above may appear in all subsets. The specific wording templates should not. For example, two notices generated from the same sentence pattern belong to the same family even when their event details differ.

Finalize the split before optimization. Keep a manifest listing IDs, family IDs, subset assignments, and a dataset hash. A hash records the frozen version; it does not prove that nobody has viewed the test data.

### 6.4 Annotation procedure

Draft notices first, then label them using the contract. Store the exact supporting phrase for each non-null field. For a null label, record whether information is absent, unresolved, or explicitly pending.

Review the labels in a second pass without looking at generated model answers. A classmate or faculty review of a small sample is useful, but not mandatory for the implementation. State who actually reviewed the dataset; do not describe solo self-checking as independent annotation.

Resolve genuine annotation ambiguity before freezing. Do not remove difficult cases after seeing which method gets them wrong.

### 6.5 Suggested record structure

```json
{
  "id": "DEV-FB-001",
  "family_id": "announcement_pending_venue_A",
  "split": "feedback",
  "category": "missing_venue",
  "notice": "The AI Club announces the Prompt Engineering Workshop. The workshop will take place on 20 October 2026. The venue will be announced later.",
  "gold": {
    "event": "Prompt Engineering Workshop",
    "date": "20 October 2026",
    "venue": null
  },
  "evidence": {
    "event": "Prompt Engineering Workshop",
    "date": "20 October 2026",
    "venue": null
  },
  "annotation_notes": {
    "venue": "Pending announcement; no location is stated."
  }
}
```

This is a documentation example, not part of a completed 60-record benchmark. Keep illustrative examples from this guide outside the final test set.

### 6.6 When the original prompt already works well

Do not deliberately corrupt model responses to create a success story. A high-performing original prompt is a legitimate result. During early development, ensure the case categories test the stated contract, then freeze them.

If every development case passes, the optimizer should return "no repair needed". The final experiment can still show the lack of additional benefit and the cost of unnecessary rewriting.

### 6.7 Optional generated test set (enabled by local throughput)

The 40 hand-written test notices give a human-authored result but a coarse one (one notice = 2.5 percentage points). With local inference, a second, larger test set is cheap to run.

- **Construction.** Write at least 20 template families (sentence patterns with slots for event, date, venue, deadline, correction, conflict). Generate notices by filling the slots; the reference answer follows from the slot assignment, so labels are correct by construction.
- **Scale (proposed).** Around 300 generated test notices, from template families that appear nowhere in the feedback, selection, or hand-written sets. Split by `family_id`, never by notice.
- **Checks.** Every non-null gold string must appear verbatim in its notice; a null case must not contain a value that the contract would count as the event date or venue. Spot-check about 10 percent by hand.
- **Reporting.** Report hand-written (Test-H) and generated (Test-G) results **separately**. Templated notices are more homogeneous and probably easier, so pooling the two would blur what each measures.

Labels that are correct by construction are not the same as human-validated labels; say which is which.

---

## 7. Architecture and module responsibilities

### 7.1 Optimization flow

```text
Task contract + original prompt + development dataset
                         |
                         v
                Run original extractor
                         |
                         v
         Local parser, schema, and reference checks
                         |
             +-----------+-----------+
             |                       |
             v                       v
    Targeted repair editor    Complete rewrite editor
             |                       |
             v                       v
       Enforce edit rules       Enforce task boundary
             |                       |
             v                       v
        Evaluate candidate on development data
                         |
                         v
         Accept or reject under fixed selection rule
                         |
                         v
          Freeze original and selected prompt versions
                         |
                         v
          Evaluate once on the 40 final test notices
                         |
                         v
              Metrics, comparisons, and report
```

The two editing methods are separate branches. Neither receives the other branch's candidate prompt or private editing history.

### 7.2 Component responsibilities

| Component | Responsibility | Requires an LLM call? |
|---|---|---|
| Dataset loader | Read records and enforce split membership | No |
| Prompt renderer | Combine sections with the current notice | No |
| Extractor | Generate event/date/venue output | Yes |
| Evaluator | Check raw JSON, schema, and reference values | No |
| Error aggregator | Count failures and select feedback examples | No |
| Localizer | Rank prompt sections by likelihood of fault (Section 9A.5) | Depends on the localizer |
| Editor | Propose one targeted patch or a complete rewrite | Yes |
| Patch guard | Check allowed component and apply the exact edit | No |
| Candidate selector | Retain the best admissible development candidate | No |
| Experiment logger | Save requests, outputs, scores, hashes, and usage | No |
| Dashboard | Display diffs, case-level results, and aggregate metrics | No |

Use ordinary Python functions for orchestration. A multi-agent framework is not necessary: the same locally served model (or a separate, larger editor model) can receive different stateless requests for extraction and editing.

### 7.3 Distinguish two operating modes

**Evaluation mode** has labelled inputs and can report correctness. **Live demonstration mode** accepts a new notice without a reference answer and can only report observable checks such as parseability, required keys, and types.

In live mode, a valid JSON object is not automatically a correct extraction. Display "reference answer unavailable" rather than an invented accuracy score.

---

## 8. Prompt design and complete templates

### 8.1 Structured representation

Store four editable sections as a dictionary: `instruction`, `context`, `constraints`, and `output_format`. The notice is a separate runtime input, not an editable part of the reusable template.

For targeted repair in the restricted baseline, only `constraints` and `output_format` are eligible; in the novelty experiments (Section 9A) any one of the four sections may be edited, chosen by a localizer. A complete rewrite may revise all four sections but may not change the output contract or the input-handling mechanism.

### 8.2 Original prompt

```text
[INSTRUCTION]
Extract details about the target college event from the supplied notice.

[CONTEXT]
The notice describes one clearly named target college event.

[CONSTRAINTS]
Keep the result concise and use the notice.

[OUTPUT_FORMAT]
Return JSON with the keys event, date, and venue.

[NOTICE]
{{NOTICE}}
```

This prompt specifies the basic task but incompletely states the null, conflict, and strict formatting rules. Preserve it exactly as the baseline after the study configuration is frozen.

### 8.3 Feedback packet

Create feedback programmatically from the 12 feedback cases. Both editors receive the same packet format: contract, current prompt, error counts, and up to four failed examples, plus up to two passing examples to help preserve correct behavior.

Choose examples by a deterministic rule: cover distinct error codes first, then use case ID ordering. Do not choose unusually favorable examples for one method.

```json
{
  "feedback_subset_size": 12,
  "error_counts": {"NULL_POLICY_VIOLATION": 1},
  "failed_examples": [
    {
      "id": "ILLUSTRATION-001",
      "notice": "The AI Club announces the Prompt Engineering Workshop on 20 October 2026. The venue will be announced later.",
      "expected": {
        "event": "Prompt Engineering Workshop",
        "date": "20 October 2026",
        "venue": null
      },
      "observed": {
        "event": "Prompt Engineering Workshop",
        "date": "20 October 2026",
        "venue": "Main Auditorium"
      },
      "errors": ["NULL_POLICY_VIOLATION"]
    }
  ]
}
```

The observed output above is hypothetical. An actual feedback packet must contain raw responses from the development run.

### 8.4 Targeted repair editor template

```text
You are editing a reusable prompt for a fixed information-extraction task.

Your job is to propose ONE local edit based on observed development errors.
Do not answer the notices yourself. Do not rewrite the entire prompt.

TASK CONTRACT:
{{TASK_CONTRACT}}

CURRENT PROMPT SECTIONS:
{{PROMPT_SECTIONS_JSON}}

DEVELOPMENT FEEDBACK:
{{FEEDBACK_JSON}}

EDIT RULES:
- Choose only constraints or output_format.
- Replace one contiguous, nonempty span occurring exactly once in that section.
- Leave every other section unchanged.
- Use no more than 60 whitespace-separated words in new_text.
- The complete resulting prompt must remain within 240 whitespace-separated words,
  excluding the notice and section labels.
- Correct the intended task; do not introduce a new task or new output fields.
- Generalize the rule. Do not insert event names, dates, venues, case IDs,
  or complete answers from the development examples.
- Provide a short evidence-based explanation, not a claim of proven causation.
- If no justified edit exists, return action = no_change.

Return one JSON object with these keys:
action, component, old_text, new_text, error_codes, explanation.

For action = replace, component must be constraints or output_format.
For action = no_change, use null for component, old_text, and new_text.
```

Example patch proposal:

```json
{
  "action": "replace",
  "component": "constraints",
  "old_text": "Keep the result concise and use the notice.",
  "new_text": "Use only explicitly stated information. Return null for an absent or unresolved date or venue. Do not guess.",
  "error_codes": ["NULL_POLICY_VIOLATION"],
  "explanation": "The feedback includes a non-null venue where the notice supplies no location."
}
```

This is a proposed patch, not a demonstrated improvement.

### 8.5 Complete rewrite editor template

```text
You are rewriting a reusable prompt for a fixed information-extraction task.

Rewrite the prompt using the task contract and observed development errors.
You may revise instruction, context, constraints, and output_format.

TASK CONTRACT:
{{TASK_CONTRACT}}

CURRENT PROMPT SECTIONS:
{{PROMPT_SECTIONS_JSON}}

DEVELOPMENT FEEDBACK:
{{FEEDBACK_JSON}}

RULES:
- Preserve the task contract and exactly the same three extraction fields.
- Keep the four named sections and the unchanged runtime notice mechanism.
- Generalize from errors; do not insert case-specific answers or case IDs.
- The complete prompt must stay within 240 whitespace-separated words,
  excluding the notice and section labels.
- Explain the intended changes briefly.
- Do not claim that the rewrite improves performance before testing it.

Return one JSON object with keys:
action, sections, explanation.

Use action = rewrite and include the four section strings in sections.
Use action = no_change and sections = null when no change is justified.
```

Both editors receive the same maximum number of attempts and the same configured response budget. Their observed token usage may still differ; equal attempt limits do not imply equal token consumption.

### 8.6 Enforce constraints in code

For a targeted patch, require the named component to be allowed, `old_text` to occur exactly once, and all unedited sections to remain identical. Apply only that replacement. Check the word limits after rendering.

Reject unknown actions, missing keys, multiple edits, empty replacement text, and attempts to modify the task input or schema. A malformed editor response consumes an editing attempt; do not grant unlimited extra chances until a preferred result appears.

A guard can flag copied case IDs and obvious example literals. It cannot prove semantic faithfulness by string matching alone. Apply the same documented human admissibility review to both methods before test evaluation, checking only contract preservation and case-specific leakage, not inventing extra prompt improvements.

### 8.7 Structured-output API settings

For the **extractor comparison**, keep provider-enforced structured output off for every method. This allows the study to measure whether prompt changes improve both content and formatting.

A separate optional experiment could enable it for every arm. Do not enable it only for targeted repair. For locally served models, the analogue is constrained decoding in the serving library (check its documentation). Google's documentation, relevant if the optional hosted-API arm is used, distinguishes syntactic structured-output guarantees from semantic correctness, which still requires application validation. [W4]

The editor's own patch-format response may use a structured-response mechanism if applied consistently to both editing methods. Record that configuration separately from the extraction settings.

---

## 9. End-to-end experimental methodology

### 9.1 Phase A: freeze the study specification

Write `task_contract.md`, prepare and review the dataset, and fix the split manifest. Choose the open-weight model(s) and record the exact Hugging Face model ID and revision, serving library and version, quantization, GPU name and VRAM, driver/CUDA versions, and generation settings.

Set the maximum editing attempts to two for each method. Fix the text-length limits, feedback-selection rule, parser behavior, and candidate-selection rule before running optimization. Do not alter these rules after seeing final test results.

### 9.2 Phase B: evaluate the original prompt

Run the original prompt once per development notice in independent requests. Save the exact prompt, source notice, raw response, request status, model metadata, and duration.

Parse and score the response locally. Do not remove introductory text, strip Markdown fences, repair quotes, or insert missing keys before the primary evaluation. Those transformations would hide the failures the project is measuring.

If a request fails because of a network or quota issue, record a transport failure separately from a model response error. Resume according to the fixed retry policy, not according to whether the answer looks correct.

### 9.3 Phase C: construct evidence-based feedback

For the 12 feedback cases, aggregate error codes and construct the packet described in Section 8. The error report should identify what was observed, for example:

```text
Case: DEV-FB-007
Field: date
Expected: null
Observed: "25 October 2026"
Error: NULL_POLICY_VIOLATION
```

This is better evidence than "the prompt is weak". However, it does not establish the psychological or internal reason the model made the error.

### 9.4 Phase D: optimize the two branches

Initialize both branches from the same original prompt and its saved development scores.

In the targeted branch, make one editor call, validate the patch, apply it, and evaluate the resulting prompt on all 20 development examples. In the rewrite branch, do the same with one complete rewrite. Each branch may make at most two editing attempts.

For a second attempt, use feedback from that branch's current retained prompt. Both branches have access to the same feedback subset and contract; their observed errors may naturally differ after the first attempt.

An inadmissible candidate is rejected without evaluation. A `no_change` decision stops that branch. A second attempt starts from the retained prompt, not from an unaccepted candidate. Save rejected candidates and explanations too.

### 9.5 Candidate-selection rule

Use a deterministic rule rather than manually selecting the prompt that looks nicest. Compare candidates lexicographically by this tuple:

```text
(
    correct complete records on the 8 selection examples,
    correct individual fields on the 8 selection examples,
    correct complete records on the 12 feedback examples,
    correct individual fields on the 12 feedback examples,
    negative character edit distance from the original prompt,
    negative rendered prompt length
)
```

Higher is better. Earlier criteria take priority over later ones. Keep the incumbent on an exact tie.

This favors held-back development performance before performance on examples shown to the editor. It uses edit distance as a tie-breaker, not as an excuse to accept worse correctness. Apply the same rule to both methods and disclose it because it can affect which candidate is chosen.

Only compare candidates whose development evaluation has completed. A temporarily interrupted run is not a low-scoring prompt to discard. The eight selection examples are small, so this rule is a practical selection policy, not proof that the chosen prompt is globally optimal.

### 9.6 Phase E: freeze the prompt versions

Save the original and selected prompts with hashes, parent relationships, attempt histories, and configuration. Once frozen, disable editing in the final-test workflow.

If a branch retains the original prompt, report "no accepted change". Do not invent a revised prompt merely to make all dashboard columns look different.

### 9.7 Phase F: final test evaluation

Run the 40 test notices under all three frozen versions. Each request contains one notice and one prompt version, without reference labels or previous conversation history.

Interleave or deterministically shuffle the order of the three methods within each notice. This reduces the risk of all of one method running during a different service condition. Save the ordering seed and the actual order.

Do not regenerate incorrect responses until they pass. Final accuracy is based on the first completed response for each registered method-case-replicate job. Transport retries follow the fixed operational policy and are logged separately.

### 9.8 Phase G: analyze, then stop optimizing

Compute metrics, inspect paired successes and failures, and prepare the report. You may describe newly discovered test errors as future work, but you may not use them to improve the prompt and still call the resulting score held-out performance on the same test set.

If a genuine evaluation-code bug is discovered, version and disclose the correction, apply it to all methods, and regenerate the affected analysis consistently. Preserve the original artifacts rather than selectively deleting inconvenient results.

### 9.9 Controller pseudocode

The following is an algorithm sketch, not the complete application:

```text
freeze contract, split manifest, model settings, and selection policy
original_scores = evaluate(original_prompt, development_20)

for method in [targeted_repair, complete_rewrite]:
    retained_prompt = original_prompt
    retained_scores = original_scores

    for attempt in [1, 2]:
        if all 20 development records are correct:
            stop this branch

        feedback = summarize only feedback_12 from retained_scores
        proposal = call editor once(method, retained_prompt, contract, feedback)
        save proposal and editing-call metadata

        if proposal.action == no_change:
            stop this branch

        if proposal violates edit rules or task contract:
            save rejection
            continue

        candidate = apply proposal to retained_prompt
        candidate_scores = evaluate(candidate, development_20)
        wait/resume until development jobs are operationally complete

        if selection_key(candidate) > selection_key(retained_prompt):
            retained_prompt = candidate
            retained_scores = candidate_scores

    freeze retained_prompt for this method

for case in final_test_40:
    for method in registered interleaved order:
        generate one response using that frozen prompt
        score locally against the case's reference answer

export results, prompt history, usage, and limitations
```

The first stopping condition refers to the **retained** prompt's scores. Every selection score and feedback packet must be associated with the correct prompt hash.

## 9A. Novelty experiments: defect injection and fault localization

This section extends Section 9 and supports contributions C1-C3 (Section 1B). It reuses the same contract, splits, evaluator and selection rule. The 14-day core runs everything marked * and all arms in Section 9A.4; the rest is Phase 2 (Section 15.4).

### 9A.1 Settings and units of analysis

A **setting** is one (model, starting prompt) pair. Each setting is an independent optimization run with its own development scores. Test notices are repeated measures *within* a setting, so claims that generalize across defects and models should treat the **setting**, not the notice, as the unit (Section 9A.6).

### 9A.2 Oracle prompt and defect catalog*

Write one **oracle prompt** from the task contract alone, before looking at any model output, using the same four sections as Section 8.2. It states the null, conflict, correction, deadline and raw-JSON rules explicitly. It is the ceiling and the source of every injected defect.

Create each defective variant by editing **exactly one section** of the oracle prompt:

| ID | Section edited | Injected defect | Expected main symptom |
|---|---|---|---|
| D1 | constraints | Delete the rule that absent or pending values become null | `NULL_POLICY_VIOLATION` |
| D2 | constraints | Delete the rule that deadlines and publication dates are not the event date | `DATE_MISMATCH` |
| D3 | constraints | Replace the correction rule with "return null whenever two values appear" (over-broad) | `OMITTED_SUPPORTED_VALUE` |
| D4 | constraints | Add a conflicting directive: "always provide a value for every field" | `NULL_POLICY_VIOLATION` |
| D5 | output_format | Weaken "exactly one JSON object and no other text" to "return JSON" | `MALFORMED_JSON` |
| D6 | output_format | Drop one required key from the listed keys | `SCHEMA_KEYS` |
| D7 | instruction | Replace the extraction instruction with a vague one ("summarize the notice") | Mixed |
| D8 | context | State wrongly that a notice may contain several events | Mixed |

These defect families follow the specification and structure/format categories of the prompt-defect taxonomy [L18]; the specific wording is this project's own design. The **naturalistic** starting prompt of Section 8.2 is kept as one more setting with no known fault location.

### 9A.3 Defect screening*

Keep a defect for a given model only if it lowers **strict record accuracy on the 20 development notices** by at least a pre-registered margin (proposed: 15 points) relative to the oracle prompt. Screen on development data only, never on test data. A defect that does not hurt a given model is dropped *for that model*, and the drop is reported.

### 9A.4 Arms*

| Arm | Description |
|---|---|
| Defective | The defective prompt, unchanged |
| Rewrite | Complete rewrite (Section 8.5), at most two attempts |
| Span edit | LPO-style: the editor marks one contiguous span anywhere in the prompt and replaces it [L5], with the same size limits as the section patch |
| Section patch + localizer *X* | One-section patch (Section 8.4) where the section is chosen by localizer *X* (Section 9A.5) |
| Oracle-section patch | Patch applied to the **true** injected section (upper bound for localization) |
| Random-section patch | Section chosen uniformly at random (lower bound) |
| Oracle prompt | The clean prompt, unchanged (ceiling) |

All editing arms share the same feedback packet, attempt limit (two), size limits, selection rule and admissibility review. In the section-patch arms **all four sections are eligible**: the editor receives the chosen section name and may edit only that one. The restricted baseline of Section 8.4 (constraints and output_format only) is kept as a reference arm.

### 9A.5 Localizers

Each localizer outputs a ranked list of the four sections for a given prompt and feedback set. The tie-break order is fixed in advance: output_format, constraints, instruction, context.

| Localizer | How it ranks sections | Needs |
|---|---|---|
| L-LLM* | The editor model reads the contract, the sections and the feedback packet and names the section(s) most likely at fault (LPO-style judgement [L5]) | One LLM call |
| L-Rule* | Maps observed error codes to sections using the table in Section 10.7, weighted by error counts | No LLM call |
| L-Ablate | Removes each section in turn and ranks by the drop in feedback-set field accuracy. Measures **importance, not fault**; included as a baseline expected to be weak | Four feedback evaluations |
| L-Probe* | Requests one patch per section, evaluates each patched prompt on the **feedback** subset only, and ranks sections by the resulting score | Four editor calls and four feedback evaluations |
| L-Grad (Phase 2) | On a white-box model, computes the loss of the reference JSON for failing feedback cases and ranks sections by gradient-based token importance aggregated per section, reported both length-normalized and raw (gradient-guided span or segment selection has precedent [L12, L13]) | Hugging Face model with gradients |

Selection data is never used by a localizer; only the acceptance rule uses it. Whether gradient saliency localizes prompt faults at all is a hypothesis to test, not an assumption.

### 9A.6 Metrics and analysis

| Metric | Definition |
|---|---|
| Top-1 localization accuracy | Fraction of settings where the top-ranked section equals the injected section |
| Mean reciprocal rank | Mean of 1/rank of the injected section |
| Restoration ratio | (accuracy after repair - defective accuracy) / (oracle accuracy - defective accuracy), on the test set; values below 0 or above 1 are reported, not clipped |
| Recovery and regression | As in Section 11.2, with the drift framing cited to [L10] |
| Edit size | Character edit distance and whitespace-word growth of the rendered template |
| Cost | Editor calls, extraction generations, and GPU time per setting |

Analysis plan: (1) within a setting, compare notices pairwise (exact McNemar test or paired bootstrap); (2) across settings, use a bootstrap or signed-rank test with the **setting** as the unit, with confidence intervals; (3) report all arms, including those that lose. With roughly 8 defects x 2-3 models the number of settings is small, so treat tests as descriptive and avoid strong language.

A localizer can pick the "wrong" section and still restore accuracy, because a different section can compensate. Report **strict** localization (matches the injected section) and **fixability** (restoration ratio above a pre-registered threshold, proposed 0.8) separately.

### 9A.7 Pre-registration*

Before the first optimization run, commit a file `analysis_plan.md` fixing: the oracle prompt, defect list and wording, screening margin, arms, localizer definitions and tie-break order, metrics, statistical tests, and seeds. Do not change it after seeing test results; record any deviation as a deviation.

### 9A.8 Phase 2 extensions

- **L-Grad** (Section 9A.5).
- **Multi-defect prompts** (two defects at once), to test whether localizers find one, both, or neither.
- **Portability** (C4): apply a repair found with model A to models B and C and measure the restoration ratio there.
- **External baselines:** an LPO-style full optimizer [L5], ProTeGi-style critique-and-edit [L3], and the public GEPA implementation [L9].
- **A second task**, to test whether findings depend on the extraction domain.

### 9A.9 Threats specific to this design

- Defects are injected by the author and so reflect the author's idea of a defect; the naturalistic setting and the taxonomy-based catalog [L18] only partly address this.
- The oracle prompt encodes the contract, so "restoration" measures a return to a contract-complete prompt, not to a universal optimum.
- The editor and the extractor may be the same model. Run the editor as a separate, larger local model if VRAM allows, and report both configurations.
- Small open-weight models can be noisy: use fixed seeds, repeated sampling where stated, and report variance.
- Templated notices (Section 6.7) are more homogeneous than real notices.

---

## 10. Worked examples and failure analysis

All examples in this section are fictional. The incorrect outputs describe possible failure modes, not measured behavior of a particular model.

### 10.1 Missing venue: content error despite valid JSON

```text
The Robotics Club announces the Robot Design Seminar on 22 October 2026.
Venue details will be shared later.
```

Reference:

```json
{"event": "Robot Design Seminar", "date": "22 October 2026", "venue": null}
```

An illustrative output containing `"venue": "Main Auditorium"` would parse correctly and have the expected keys, but it would violate the null policy. A constraints edit could explicitly prohibit guessing missing values.

This example shows why JSON validity must not be reported as extraction accuracy.

### 10.2 Registration deadline mistaken for event date

```text
Register for the Data Science Symposium by 18 October 2026.
The symposium will be held on 24 October 2026 in Seminar Hall B.
```

Reference:

```json
{"event": "Data Science Symposium", "date": "24 October 2026", "venue": "Seminar Hall B"}
```

An output date of `18 October 2026` appears in the source, but has the wrong role. A simple string-overlap check cannot distinguish the event date from the deadline.

A candidate constraint could say: "Extract the event date, not registration deadlines or notice publication dates." Its value must be tested on the remaining development examples, not assumed from this one case.

### 10.3 Correct data with forbidden surrounding text

```text
Here is the extracted information:
{"event": "Data Science Symposium", "date": "24 October 2026", "venue": "Seminar Hall B"}
```

This fails the required raw-output contract. A local parser must not silently discard the first line in the primary evaluation.

The likely repair location is the output-format section: "Return exactly one JSON object and no other text." Content rules need not change in that attempt.

### 10.4 An explicit correction is not unresolved ambiguity

```text
The Applied AI Workshop was originally scheduled for 21 October 2026.
It has now been rescheduled to 26 October 2026. Venue: Innovation Lab.
```

Reference:

```json
{"event": "Applied AI Workshop", "date": "26 October 2026", "venue": "Innovation Lab"}
```

A rule that says "return null whenever two dates appear" would be too broad. The notice resolves which date is current. A repair must distinguish an explicit correction from an unresolved contradiction.

### 10.5 Unresolved venue conflict

```text
The Cloud Computing Seminar will be held on 27 October 2026.
One section of the notice lists Seminar Hall A as the venue.
Another section lists Seminar Hall C, and no correction or priority is stated.
```

Reference:

```json
{"event": "Cloud Computing Seminar", "date": "27 October 2026", "venue": null}
```

Selecting either hall violates this project's conflict policy. That does not mean the selected hall was fabricated: both appear in the notice. Describe the error as **unwarranted field filling under unresolved conflict**, not automatically as hallucination.

### 10.6 Regressions matter

A new instruction may fix the unresolved-conflict example but incorrectly return null for the explicit-correction example. That is a regression.

The report should show both directions: cases repaired and cases newly broken. A higher overall score does not imply that every individual case improved.

### 10.7 Error taxonomy

| Error code | Observable condition | Likely repair location |
|---|---|---|
| `MALFORMED_JSON` | Raw response cannot be strictly parsed | Output format |
| `SCHEMA_KEYS` | Missing or additional keys | Output format |
| `WRONG_TYPE` | Value has a disallowed type | Output format |
| `EMPTY_STRING` | A required string is empty or whitespace-only | Constraints or output format |
| `EVENT_MISMATCH` | Event value differs from the reference | Constraints |
| `DATE_MISMATCH` | Event date differs from the reference | Constraints |
| `VENUE_MISMATCH` | Venue differs from the reference | Constraints |
| `NULL_POLICY_VIOLATION` | Non-null output where the contract requires null | Constraints |
| `OMITTED_SUPPORTED_VALUE` | Null output where a supported value exists | Constraints |

These are diagnosis aids, not a proven mapping from every model failure to a unique cause. API outages, rate limits, and request timeouts belong in operational status fields, not this prompt-error taxonomy.

---

## 11. Metrics and interpretation

### 11.1 Primary metric: strict record accuracy

A record is correct only when the raw response parses, matches the schema, and all three values match the reference under the predefined normalization rule.

```text
Strict record accuracy = correct complete records / registered test notices
```

For a complete main run, the denominator is 40 for each method. A record with one wrong field is not a strict success.

### 11.2 Secondary metrics

| Metric | Definition | Purpose |
|---|---|---|
| JSON validity | Strictly parseable responses / 40 | Isolate raw-format failures |
| Schema validity | Schema-valid responses / 40 | Check required keys and types |
| Field accuracy | Correct required fields / 120 | Show partial extraction correctness |
| Null-policy accuracy | Correctly returned nulls / reference-null fields | Evaluate absent or unresolved information |
| Unwarranted filling | Predicted non-null values / reference-null fields | Detect guessing or unresolved assignments |
| Recovery rate | Original failures corrected / original failures | Measure repair benefit |
| Regression rate | Original successes broken / original successes | Measure damage to existing performance |
| Character edit distance | Levenshtein distance between rendered templates | Quantify how much text changed |
| Prompt growth | Final minus original prompt length | Show expansion in characters or words |
| Usage and latency | Logged calls, provider-reported tokens, and durations | Measure practical overhead |

When a denominator is zero, report `N/A`, not zero percent. For field accuracy, unparseable outputs earn zero correct fields. An output with extra keys may have correct individual fields but still fails schema and strict record accuracy.

For unwarranted filling, count an explicit, non-null predicted value in a required-null field. Keep unparseable responses separate; do not pretend every parse failure is an invented field. Missing keys also fail null-policy accuracy even though they are not explicit non-null values.

### 11.3 Distinguish null-policy violations from hallucinations

A value can be wrong without being absent from the source. Registration dates and conflicting venues illustrate this distinction.

Use the mechanical metric name **unwarranted filling** for the main experiment. A separate manually audited count of fabricated values is possible, but it requires checking source evidence and documenting the auditing rule. Do not call all incorrect values hallucinations.

### 11.4 Paired comparison

Because all methods see the same notices, compare outcomes case by case. The following counts are **hypothetical arithmetic only**:

| Original outcome | Targeted outcome | Illustrative count |
|---|---|---:|
| Correct | Correct | 26 |
| Correct | Incorrect | 4 |
| Incorrect | Correct | 8 |
| Incorrect | Incorrect | 2 |

These hypothetical counts give original accuracy `30/40 = 75%`, targeted accuracy `34/40 = 85%`, recovery `8/10 = 80%`, and regression `4/30 = 13.3%`. The net gain is **10 percentage points**, not 10 percent relative improvement.

This example demonstrates why reporting only the gain would hide four newly broken cases. Replace it with actual counts in the project report.

### 11.5 Editing effort and token accounting

Measure edits on the rendered reusable template, excluding the changing notice. Specify whether an edit-distance value counts characters or words; do not label a similarity ratio from `difflib` as Levenshtein distance.

Whitespace word counts enforce the proposed editing limits. They are **not model-token counts**. Record actual token counts (from the serving library or tokenizer, or provider-reported usage if the optional hosted-API arm is used) and GPU time separately, without double-counting categories already included in totals.

Measure optimization usage separately from final inference usage. A prompt that is shorter at deployment may still have required more calls to discover. No cost saving should be claimed without including that setup cost.

### 11.6 Small-sample interpretation

One notice represents 2.5 percentage points on a 40-notice test set. Treat small differences cautiously. This is a descriptive pilot with one model, one task, one initial prompt, and one main extraction response per case.

Optional repeated runs or paired uncertainty analysis can strengthen a later study, but do not inflate the independent sample size by treating three fields from the same notice as three independent notices.

### 11.7 Results table to populate after execution

| Method | Strict correct / 40 | JSON-valid / 40 | Field-correct / 120 | Recovered cases | Regressed cases | Edit distance | Total calls |
|---|---|---|---|---|---|---|---|
| Original | Not run | Not run | Not run | N/A | N/A | 0 | Not run |
| Complete rewrite | Not run | Not run | Not run | Not run | Not run | Not run | Not run |
| Targeted repair | Not run | Not run | Not run | Not run | Not run | Not run | Not run |

No performance result is implied by the title or by the examples in this guide. The expanded table for the novelty arms (localization accuracy, restoration ratio) is defined in Section 9A.6 and must likewise stay empty until the experiments are run.

---

## 12. Fairness, reproducibility, and limitations

### 12.1 Variables that must stay fixed

Keep the model identifier, extraction generation settings, source notices, output contract, parser, normalizer, feedback-subset access, and maximum editing attempts identical across comparable methods.

Use stateless extraction requests. Do not let a later method benefit from a previous method's answers or from a shared chat history. Also keep tools, retrieval, native structured output, and automatic answer repair disabled in the main extractor comparison.

Low temperature is not a guarantee of identical outputs. Save the chosen settings and model/version metadata rather than claiming complete determinism.

### 12.2 What is intentionally different

The intervention is the edit policy: local repair of one eligible component versus a rewrite of all prompt sections. Different prompt lengths and token usage are outcomes to measure, not differences to hide.

The attempt budget is matched. Token budgets are capped consistently but actual consumption is not guaranteed equal. Describe the experiment as **attempt-capped**, not exactly equal-compute, unless you enforce and document a stricter compute-matching procedure.

### 12.3 Candidate and test leakage

The editor may use feedback examples. It may not receive selection-example content or final-test examples and labels. A human preparing the dataset will necessarily see its contents; disclose that the split is an experimental separation, not a technical guarantee of human blindness.

Never tune the prompt, model settings, evaluator, or admissibility rules to favor a method after examining final results.

### 12.4 Useful limitations to acknowledge

The artificial domain and small test set limit generalization. One starting prompt cannot establish that local editing universally beats rewriting. The editor and extractor may be the same model, so findings are specific to that configuration. Rule-based evaluation is practical because the task has a narrow contract; it is not automatically suitable for open-ended essays or subjective quality judgments.

An experiment can support "targeted repair performed this way on these notices". It cannot establish a new state-of-the-art method or conference-level novelty from this pilot alone.

### 12.5 Honest negative outcomes

If the rewrite wins, explain which changes helped it. If all methods tie, report whether the original was already sufficient. If targeted repair uses fewer edits but lower accuracy, show the tradeoff rather than declaring a win on efficiency alone.

A completed, reproducible evaluation with a negative result still demonstrates the intended prompt-engineering skills.

---

## 13. Application design and project structure

### 13.1 Main screens

**Prompt workspace.** Show the four prompt sections, fixed contract, model configuration, and editing limits. Display a visible distinction between development mode and frozen final-test mode.

**Development comparison.** Show the original response, expected answer, error codes, editor proposal, highlighted diff, and candidate acceptance decision. Retain rejected revisions in history.

**Evaluation dashboard.** Show the three methods side by side, aggregate metrics, paired recoveries/regressions, case-level records, operational failures, and actual usage. Include an export action.

**Live demo.** Accept a new notice and use a selected frozen prompt. Show schema checks, but no correctness percentage without a human reference.

These can be sections of one marimo app rather than separate frontend projects.

### 13.2 Suggested repository layout

```text
promptrepair/
    app.py                      # marimo app (run on molab)
    notebooks/                  # marimo notebooks: setup, sweeps, analysis
    analysis_plan.md            # pre-registered plan (commit before sweeps)
    README.md
    task_contract.md
    requirements.txt
    requirements.lock.txt
    .gitignore
    config/
        experiment.json
        output_schema.json
    data/
        feedback.jsonl
        selection.jsonl
        test_inputs.jsonl
        test_labels.jsonl
        generated_test_inputs.jsonl
        generated_test_labels.jsonl
        split_manifest.json
    prompts/
        original.json
        oracle.json
        defects/
        targeted_editor.txt
        rewrite_editor.txt
        frozen/
    src/
        inference.py            # local model adapter (optional API adapter)
        defects.py              # oracle prompt and defect injection
        localizers.py           # L-LLM, L-Rule, L-Ablate, L-Probe, L-Grad
        prompt_renderer.py
        evaluator.py
        diagnostics.py
        patcher.py
        optimizer.py
        experiment_runner.py
        reporting.py
        storage.py
    tests/
        test_evaluator.py
        test_patcher.py
        test_splits.py
        test_runner.py
        test_defects.py
        test_localizers.py
    runs/
        RUN_ID/
            manifest.json
            editor_history.jsonl
            responses.jsonl
            metrics.json
            paired_results.csv
            report.md
```

Keep reference labels out of the extraction-request construction path. Separating `test_inputs.jsonl` and `test_labels.jsonl` helps enforce that boundary in code, although filenames alone are not a protection mechanism.

### 13.3 Model adapter contract

Hide provider-specific SDK details behind a small function that accepts a rendered prompt and returns a structured record containing response text, status, model metadata, usage, and duration.

The rest of the application should not need to know whether a response came from a locally served model or an optional hosted API. The default adapter batches requests to a locally loaded open-weight model and records model ID and revision, serving library and version, GPU name, seed, and token counts. An optional hosted-API adapter (Google's getting-started guide uses the `google-genai` package [W3]) can sit behind the same interface for a stronger-editor control (Section 14.9).

Use independent requests for this experiment, without linking a previous interaction. The project does not need persistent model-side conversation state.

### 13.4 Run logging

Each generation record should contain `run_id`, subset, case ID, method, prompt hash, model ID, generation settings, replicate number, request timestamps, raw output, request status, retry count, cache status, finish metadata, and usage where available. Also record model revision, serving library and version, quantization, GPU name, seed, and a setting ID (model, starting prompt).

Store evaluation flags beside the raw output rather than replacing it with a cleaned answer. Keep editor calls in a separate log so optimization and extraction costs can be distinguished.

### 13.5 Environment on marimo molab

These steps prepare the environment; they do not create the application. Details change, so confirm each one against current molab documentation on Day 1 [M1, M3].

1. Create a molab notebook with the GPU option, run `nvidia-smi` in a cell, and record the GPU name, memory, and driver/CUDA versions in `config/experiment.json`.
2. Declare packages in the notebook. molab uses marimo's package sandboxing, which records dependencies in the notebook file itself [M4]. Typical needs: `torch` (often preinstalled), `transformers`, `accelerate`, `pytest`, and optionally a serving library such as `vllm`. If a serving library conflicts with the preinstalled stack, fall back to batched generation with Transformers.
3. Download one open-weight instruct model from Hugging Face and run a smoke test. Record the exact model ID and revision. Some models are gated and need a token or licence acceptance; **never paste a token into a notebook**, because molab notebooks are public but not discoverable by default [M3].
4. Keep the core pipeline as plain Python modules in a GitHub repository and mirror it into molab; molab can synchronize notebooks from GitHub, with GitHub as the source of truth [M1]. The marimo notebooks (`app.py`, `notebooks/`) are thin front ends that call `src/`.
5. Persist run logs outside the session: write `runs/RUN_ID/` to Google Drive (molab notebooks can read and write Drive files [M2]) or push to GitHub after each batch. Do not rely on the session's local disk surviving.
6. Optionally, `uvx marimo edit <notebook-url>` opens a molab notebook on your own machine [M3]; the GPU parts will not run without a local GPU.

Run the finished app with `marimo run app.py`, or share it from molab as an app or as slides [M1, M4].

---

## 14. GPU compute on molab, throughput budget, and resilience

### 14.1 What is and is not confirmed about molab

Confirmed by the molab pages reviewed on 19 October 2026: notebooks can run on GPU compute on CoreWeave infrastructure in a free public preview; long-running notebooks can run for up to 12 hours before molab shuts them down; notebooks can be mirrored from GitHub; notebooks can read and write Google Drive files; and notebooks are public but not discoverable by default [M1-M3].

Not confirmed by those pages: the GPU model and VRAM (you state an RTX 6000-class card), disk size and persistence, outbound network access to model hubs, per-user quotas, and how long the free preview will last.

**Day-1 checklist:** run `nvidia-smi`; check free disk; download a model from Hugging Face; install your serving library; write to Drive or push to GitHub; measure tokens per second on a 200-prompt batch.

### 14.2 Model choice

Pick the open-weight instruct models on Day 1 from what is currently on Hugging Face, because versions change quickly; this document deliberately names none. Prefer two or three models from different families at roughly 7B-14B parameters. As arithmetic only: bf16 weights take about 2 bytes per parameter (a 14B model is roughly 28 GB before KV cache and overhead), so check VRAM before choosing and use quantization if needed. Check each model's licence terms and record model IDs and revisions in the report.

If VRAM allows, serve a separate, larger model as the **editor** and report editor-extractor identity as a variable; otherwise use the same model for both and say so.

### 14.3 Serving and what each route gives you

| Need | Practical route |
|---|---|
| Fast batched extraction | A serving library with continuous batching (for example vLLM), if it installs cleanly; otherwise Transformers batched generation |
| Token log-probabilities | Either route (verify the option in the library documentation) |
| Gradients for L-Grad | Hugging Face Transformers with a model loaded for a forward/backward pass; not a serving library |

This document has not tested any of these on molab; verify installation and API details in current documentation.

### 14.4 Job budget and session time

Count generations per setting (one model, one starting prompt) with the 12/8/40 splits and two attempts per method:

| Activity | Calculation |
|---|---|
| Starting prompt on development | 20 |
| Candidate development evaluations | arms x attempts x 20 |
| Probe and ablation localization | about 4 sections x 12 feedback notices per attempt, for each localizer that uses them |
| Final test | frozen prompts x test notices x S (S = sampled repeats; S = 1 if greedy only) |

Proposal-only example: 8 editing arms (rewrite, span edit, four localizer arms, oracle-section, random-section), 2 attempts, 20 development notices gives 320 candidate evaluations; probe and ablation localization adds roughly 150; final test with 10 frozen prompts (8 arms + defective + oracle) x 340 test notices (40 hand-written + about 300 generated) x S = 3 gives 10,200. That is about 10,700 generations per setting. With 3 models x 8 defects = 24 settings the full sweep is on the order of 10^5 short generations. Whether that fits depends entirely on measured throughput.

**Rule:** measure throughput on Day 1, compute the sweep time, and design each sweep to finish well inside the 12-hour session limit [M1] (a proposed target is at most 6 hours), checkpointing after every batch so a shutdown loses little work. Reduce repeats, test notices, or settings before compromising the integrity of the final test.

### 14.5 Determinism

Use greedy decoding (temperature 0) for the primary comparison where the library supports it, and record seeds. Batched GPU inference may not be bitwise reproducible across batch compositions or software versions, so do not claim exact determinism. For sampled arms, use fixed seeds and report the variation across them.

### 14.6 Cache and resume

Build the cache key from model ID and revision, serving library version, quantization, generation settings, complete prompt text, notice content, and seed. A cached result is a replay, not a new sample: mark it and exclude it from timing summaries. If two arms end with identical prompt hashes, disclose that responses were reused. Increment the replicate ID for deliberately repeated trials.

### 14.7 Failure handling

Record operational failures (out-of-memory, session shutdown, download failure) separately from model errors. Retry transient failures a bounded number of times (proposed: two), with a smaller batch after an out-of-memory error. Never retry a completed response because its content is wrong. If a sweep is interrupted, resume from the checkpoint using the same registered jobs, and report any incomplete denominators.

### 14.8 Data handling and public notebooks

Use only fictional notices. Because molab notebooks are public but undiscoverable by default [M3], keep tokens, keys, and private data out of notebooks. Keep any gated-model token in whatever secret mechanism molab provides (confirm that one exists), or avoid gated models.

### 14.9 Optional hosted-API arm

The earlier draft used a hosted API. It can be kept as an **optional** control (for example, a stronger hosted editor) behind the same adapter. If used, track its calls separately, check the provider's free-tier availability and data-use terms [W1, W2], and never mix hosted and local results in one headline number. Do not rotate keys to bypass limits.

---

## 15. Fourteen-day implementation plan

The schedule assumes basic Python familiarity, several focused hours per day, and a working molab GPU session. It is a planning estimate, not a guarantee. **Days 1-14 deliver the core, including the novelty experiments marked * in Section 9A (C1, C2 with L-LLM / L-Rule / L-Probe, and C3).** The rest is Phase 2 (Section 15.4).

| Day | Main work | Concrete checkpoint |
|---|---|---|
| 1 | molab GPU check, install serving library, download one model, smoke test, measure throughput, set up GitHub and Drive persistence | One real batched generation; recorded GPU/VRAM and tokens per second |
| 2 | Finalize contract; write the **oracle prompt**, defect catalog D1-D8, and the naturalistic original prompt; dataset blueprint | Frozen specification; draft `analysis_plan.md` |
| 3 | Write the 60 hand-written notices; write template families for the generated set; assign `family_id` and splits | Reviewed labels and split manifest |
| 4 | Evaluator, offline unit tests, defect-injection code and tests | Known fixtures scored correctly; injector changes exactly one section |
| 5 | Local inference adapter, batching, cache, resumable runner; oracle and defective development runs; defect screening | Screening table per model; defects kept or dropped |
| 6 | Diagnostics, patch guard, rewrite editor, span editor | One accepted and one rejected patch handled correctly |
| 7 | Localizers L-LLM, L-Rule, L-Probe (and L-Ablate if time), oracle/random-section arms, selection policy; **commit `analysis_plan.md`** | Console pipeline runs one full setting end to end |
| 8 | Development sweeps for all settings; freeze prompts | Immutable frozen prompts and complete edit histories |
| 9 | Final-test sweeps (Test-H and Test-G) | Saved paired outputs and progress tracking |
| 10 | Finish sweeps; inspect operational failures | Complete registered job set, no duplicates |
| 11 | Metrics: localization accuracy, restoration ratio, recovery/regression, confidence intervals | Tables and paired results |
| 12 | Failure analysis; marimo dashboard over saved results | Honest results and limitations section |
| 13 | Report draft, presentation, recorded replay demo | Rehearsed explanation |
| 14 | Fix non-experimental bugs; package repository, logs, and report | Submission-ready repository |

### 15.1 Critical path

**Contract -> oracle prompt and defects -> labels -> evaluator -> screening -> development sweeps -> pre-registration commit -> prompt freeze -> final sweeps.** An attractive dashboard cannot compensate for a missing step.

The largest risks are molab setup surprises (install conflicts, model download, session limits), incorrect labels, throughput lower than assumed, and scope growth.

### 15.2 Scope reduction order

First drop visual polish and the dashboard. Then drop the span-edit arm and L-Ablate. Then reduce to one model, then to fewer defects (keep at least one per section), then reduce repeats. Keep the pre-registration, reviewed labels, the separate final test, raw-response logging, and the oracle/random bounds. If you shrink the study, change the denominators everywhere and say so; do not present a partial run as the full plan.

### 15.3 Practical completion target

Aim for a working console pipeline on one setting by Day 7. The marimo app should call that pipeline rather than contain experimental logic.

### 15.4 Phase 2 (weeks 3-6, optional, paper-oriented)

| Week | Work |
|---|---|
| 3 | L-Grad localizer on a Transformers model; multi-defect prompts |
| 4 | Second and third models; portability experiment (C4) |
| 5 | External baselines: LPO-style optimizer, ProTeGi-style critique-and-edit, GEPA's public code |
| 6 | Optional second task; independent annotation of a sample; full literature re-check; write-up |

---

## 16. Testing and acceptance criteria

### 16.1 Evaluator tests

Test valid JSON, missing keys, extra keys, wrong types, string `"null"`, genuine null, duplicate keys, Markdown fences, explanatory prefixes, wrong dates, empty strings, and values supported by the source but assigned to the wrong field.

The evaluator must give identical results for identical raw text and labels. It must not perform hidden cleanup that changes the scientific comparison.

### 16.2 Patch-guard tests

A valid targeted edit must alter exactly one permitted component. Reject an instruction edit, an old span that occurs twice, an absent old span, an oversized replacement, or a candidate that introduces a new field.

Test that rejected patches leave the current prompt unchanged and still count against the editing-attempt limit.

### 16.3 Pipeline tests

Use mocked model responses to verify the control flow without spending GPU time: correct prompt-hash assignment, branch isolation, subset boundaries, attempt caps, no-change behavior, candidate rejection, acceptance, caching, and resumption.

Mock tests verify software behavior only. Keep their run IDs and files separate from real-model evaluation records.

### 16.4 Completion criteria

The project is ready when it can load the reviewed data, evaluate the original prompt, propose and enforce a targeted edit, generate a complete-rewrite comparator, select candidates without test leakage, and produce a case-level final comparison with traceable raw outputs.

It must also distinguish live calls from cached replays, show operational failures, export the frozen prompt versions, and explain its limitations in the presentation.

**A specific improvement percentage is not a completion criterion.** Functionality and a valid experiment are controllable; the model's measured outcome is not.

### 16.5 Minimum evidence package

The submission should contain the frozen contract and split manifest, exact prompts, labelled synthetic dataset, evaluator tests, raw run logs, summary metrics, paired failure analysis, and a short demo recording.

Do not include API keys. Do not replace logs with screenshots alone.

### 16.6 Tests for the novelty components

- The defect injector changes **exactly one** section and leaves every other character of the oracle prompt unchanged (compare section by section).
- Screening uses development data only; assert that no test notice is reachable from the screening code path.
- Localizer outputs are deterministic given the same inputs and tie-break order. Test the metric code (top-1, MRR, restoration ratio) on hand-built rankings, including ratios below 0 and above 1 and a zero denominator (report N/A).
- Oracle-section and random-section arms use the registered seed for the random choice.
- Cache keys change when the model revision, serving library version, quantization, or seed changes.
- Resume test: kill a sweep midway, restart it, and confirm that no job is duplicated or skipped.

---

## 17. Presentation, report, and viva preparation

### 17.1 Presentation narrative

| Part | What to show |
|---|---|
| Problem | A valid-looking response can still contain wrong or invented details |
| Research question | Does a local prompt edit work as well as or better than a full rewrite? |
| Method | Component representation, evidence-based feedback, edit guard, and retesting |
| Novelty | Defect injection with a known fault location; comparison of localizers and edit granularities (Sections 1B, 9A) |
| Experiment | Local open-weight model(s), injected-defect settings, several editing arms, 20 development notices, 40 hand-written and about 300 generated test notices |
| Demonstration | Original failure, proposed patch, visible diff, and a separate notice |
| Results | Real metrics, recovered cases, regressed cases, and actual usage |
| Conclusion | What the pilot supports, what it does not, and the next experiment |

Do not show a development example as the only evidence of generalization. Make the unseen-test boundary visible in the presentation.

### 17.2 Suggested project abstract

> PromptRepair is a prompt-engineering study of targeted prompt repair for structured information extraction. Using locally served open-weight language models on a single GPU, it extracts event names, dates, and venues from fictional college notices and scores prompts with a strict rule-based checker. Starting from a contract-complete oracle prompt, the study injects one known defect into one prompt section, so that the faulty section is known by construction. It then compares complete rewriting, span-level editing, and one-section patches chosen by several fault localizers (an LLM judgement, an error-code rule table, and a probe-and-select procedure), against random-section and oracle-section bounds. The planned outcomes are localization accuracy, the fraction of the oracle prompt's accuracy that each repair restores, regressions on previously correct cases, and editing effort, measured on separate hand-written and generated test sets. The design builds on prior work in prompt optimization, including local and segment-level editing, and claims only a controlled, ground-truth comparison, not a new state-of-the-art optimizer.

After executing the project, change the abstract's tense where appropriate and insert only measured findings.

### 17.3 Questions you should be able to answer

**Why is this prompt engineering rather than model training?**
The intervention changes instructions supplied to a fixed, locally served open-weight model. No model weights are updated. The implementation evaluates how those prompt changes affect a defined task.

**What is the difference between repairing a prompt and repairing an answer?**
Answer repair changes one response. Prompt repair changes the reusable instructions and is evaluated on other inputs. The main test never asks the model to fix an answer using its hidden reference label.

**Why not simply ask the model to rewrite everything?**
That is the baseline being tested. The project does not assume it is inferior. It compares correctness and editing effort under the same development access and attempt cap.

**How do you know the changed component caused the improvement?**
The edit restriction identifies what changed in the intervention, but a small one-model experiment does not establish universal causation. We report the observed effect under fixed conditions and acknowledge model variability.

**Why use only one task?**
It makes annotation, evaluation, and implementation feasible within two weeks. It also limits the conclusion to the chosen extraction domain.

**Why not use an LLM to grade the answers?**
The target values and schema can be checked directly against reviewed labels. This avoids relying on the editor's opinion of its own work and reduces additional model calls.

**Does valid JSON mean the answer is correct?**
No. It can have the right keys and types while containing the wrong event date or an unsupported venue. That is why strict record accuracy and field correctness are separate from format validity.

**How is data leakage prevented?**
Only the feedback subset is sent to the editors. Selection data is used locally for candidate choice. Test data is evaluated after prompts are frozen and is not used for further prompt changes.

**What happens if targeted repair performs worse?**
We report that outcome, identify the failing cases, and discuss why broader rewriting may have helped. Negative results do not invalidate a properly executed comparison.

**What does the GPU change, and is any model trained?**
No model is trained or fine-tuned. The GPU is used for inference with open-weight models served in a molab session. That removes API quotas, allows several models and repeated runs, and exposes log-probabilities and gradients. The cost is that results describe those open-weight models and depend on molab availability.

**How is this different from Local Prompt Optimization, ProTeGi, or GEPA?**
LPO [L5] already restricts edits to LLM-chosen regions, ProTeGi [L3] edits prompts from error-derived feedback, and GEPA [L9] evolves prompts reflectively. This project does not claim a new optimizer. Its contribution is a protocol in which the faulty section is known by construction, so localizers and edit granularities can be compared against ground truth under matched budgets.

**Isn't injecting defects artificial?**
Partly, and that is a stated limitation. The defect catalog follows a published taxonomy [L18], and the naturalistic vague prompt is kept as an additional setting. Injection is what makes the true fault location known.

**Why report recovery and regression if others already do?**
They are standard concerns in prompt-drift work [L10, L11]. They are reported here with citation, not as a new metric.

### 17.4 Suggested report organization

Use an introduction, source-derived project basis, problem statement, objectives, scope, architecture, task and dataset specification, methodology, implementation, evaluation, results, limitations, conclusion, and references.

In the methodology, explicitly identify the changes made to fit the original PDF proposal into the two-week pilot. In the results, show the complete three-method comparison and at least one regression or explain that none was observed in this run.

---

## 18. Reference evaluator

### 18.1 Purpose and limits

The following is a complete small evaluator for the three-field contract, plus a character edit-distance function. It is **not the full PromptRepair application**. The inference adapter, optimizer, patch guard, runner, localizers, and marimo interface still need to be implemented.

It uses Python's standard library. Python documents hooks for rejecting duplicate keys and nonstandard numeric constants; the evaluator deliberately uses them rather than relying on permissive defaults. [W6]

Here, `json_valid` means compliance with the project's strict parsing policy, including unique keys. Duplicate keys are rejected by project policy; this is stricter than merely checking that JSON syntax can be decoded.

Save this as `src/evaluator.py`:

```python
from __future__ import annotations

import json
from typing import Any

FIELDS = ("event", "date", "venue")
MISSING = object()


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonstandard JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _schema_issues(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["TOP_LEVEL_NOT_OBJECT"]
    issues: list[str] = []
    if set(value) != set(FIELDS):
        issues.append("SCHEMA_KEYS")
    for field in FIELDS:
        if field not in value:
            continue
        item = value[field]
        if item is None and field != "event":
            continue
        if not isinstance(item, str):
            issues.append("WRONG_TYPE")
        elif not item.strip():
            issues.append("EMPTY_STRING")
    return list(dict.fromkeys(issues))


def _matches(actual: Any, expected: str | None) -> bool:
    if actual is MISSING:
        return False
    if expected is None:
        return actual is None
    if not isinstance(actual, str):
        return False
    # Whitespace normalization only: preserve spelling, case, and date wording.
    return " ".join(actual.split()) == " ".join(expected.split())


def score_response(raw: str, gold: dict[str, str | None]) -> dict[str, Any]:
    """Score one completed model response, without cleaning or repairing it.

    Transport failures should be recorded by the runner separately.
    """
    if not isinstance(raw, str):
        raise TypeError("raw must be a string")
    if _schema_issues(gold):
        raise ValueError("Reference answer violates the project schema")

    result: dict[str, Any] = {
        "json_valid": False,
        "schema_valid": False,
        "field_correct": {field: False for field in FIELDS},
        "record_correct": False,
        "issues": [],
        "parsed": None,
    }
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, RecursionError) as exc:
        result["issues"] = ["MALFORMED_JSON"]
        result["parse_error"] = str(exc)
        return result

    result["json_valid"] = True
    result["parsed"] = parsed
    issues = _schema_issues(parsed)
    result["schema_valid"] = not issues

    if isinstance(parsed, dict):
        for field in FIELDS:
            actual = parsed.get(field, MISSING)
            correct = _matches(actual, gold[field])
            result["field_correct"][field] = correct
            if correct or actual is MISSING:
                continue
            if gold[field] is None:
                issues.append("NULL_POLICY_VIOLATION")
            elif actual is None:
                issues.append("OMITTED_SUPPORTED_VALUE")
            else:
                issues.append(f"{field.upper()}_MISMATCH")

    result["record_correct"] = (
        result["schema_valid"] and all(result["field_correct"].values())
    )
    result["issues"] = list(dict.fromkeys(issues))
    return result


def levenshtein_chars(left: str, right: str) -> int:
    """Return character insertion/deletion/substitution distance."""
    if not isinstance(left, str) or not isinstance(right, str):
        raise TypeError("Both arguments must be strings")
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]
```

### 18.2 Unit tests

Save this as `tests/test_evaluator.py`. Run from the project root using `python -m pytest` in the virtual environment.

```python
import json
import pytest

from src.evaluator import levenshtein_chars, score_response

GOLD = {
    "event": "AI Workshop",
    "date": "20 October 2026",
    "venue": None,
}


def test_correct_response():
    score = score_response(json.dumps(GOLD), GOLD)
    assert score["record_correct"]
    assert score["schema_valid"]
    assert sum(score["field_correct"].values()) == 3


def test_missing_key_is_not_null():
    raw = json.dumps({"event": "AI Workshop", "date": "20 October 2026"})
    score = score_response(raw, GOLD)
    assert not score["record_correct"]
    assert not score["field_correct"]["venue"]
    assert "SCHEMA_KEYS" in score["issues"]


def test_extra_key_fails_record_but_preserves_field_diagnostics():
    score = score_response(json.dumps({**GOLD, "organizer": "AI Club"}), GOLD)
    assert score["json_valid"]
    assert not score["schema_valid"]
    assert not score["record_correct"]
    assert all(score["field_correct"].values())


@pytest.mark.parametrize("venue", ["null", "Main Hall", "", 0, False])
def test_non_null_value_is_not_null(venue):
    score = score_response(json.dumps({**GOLD, "venue": venue}), GOLD)
    assert not score["record_correct"]
    assert not score["field_correct"]["venue"]


@pytest.mark.parametrize("raw", [
    'Here is the result: {"event":"AI Workshop","date":null,"venue":null}',
    '```json\n{"event":"AI Workshop","date":null,"venue":null}\n```',
    '{"event":"AI Workshop","date":null,"venue":null,"venue":"Hall A"}',
    '{"event":"AI Workshop","date":NaN,"venue":null}',
    '',
])
def test_strict_parsing(raw):
    score = score_response(raw, GOLD)
    assert not score["json_valid"]
    assert not score["record_correct"]


def test_top_level_array_is_json_but_not_our_schema():
    score = score_response(json.dumps([GOLD]), GOLD)
    assert score["json_valid"]
    assert not score["schema_valid"]


def test_wrong_date_is_semantic_error():
    score = score_response(json.dumps({**GOLD, "date": "18 October 2026"}), GOLD)
    assert score["schema_valid"]
    assert not score["record_correct"]
    assert "DATE_MISMATCH" in score["issues"]


def test_whitespace_normalization_only():
    score = score_response(json.dumps({**GOLD, "event": " AI   Workshop "}), GOLD)
    assert score["record_correct"]
    changed_case = score_response(json.dumps({**GOLD, "event": "ai workshop"}), GOLD)
    assert not changed_case["record_correct"]


def test_invalid_gold_is_rejected():
    with pytest.raises(ValueError):
        score_response("{}", {"event": "AI Workshop"})


def test_edit_distance():
    assert levenshtein_chars("", "abc") == 3
    assert levenshtein_chars("prompt", "prompt") == 0
    assert levenshtein_chars("kitten", "sitting") == 3
```

The tests demonstrate evaluator behavior on authored fixtures. They do not demonstrate that any prompt or model achieves a particular accuracy. Apply a reasonable response-size limit in the API runner before passing arbitrary input to the parser.

---

## 19. Final deliverables and extensions

### 19.1 Deliverables for the two-week version

Deliver a marimo/molab application plus console pipeline, the oracle prompt and defect catalog, the committed analysis plan, the reviewed fictional dataset, frozen prompt versions, editing history, automated evaluator with tests, raw run logs, a comparison report, and presentation materials.

A Markdown methodology document alone is the planning artifact; the code and real-model evaluation are the implementation deliverables still to be completed.

### 19.2 Extensions only after the core submission

Possible later experiments include multiple initial prompts, a second task, another model, independent annotation, repeated generations, a native structured-output control, or a more complete component-ablation study.

The carefully written oracle prompt (Section 9A.2) is now part of the core design as the ceiling arm; it tests whether the optimizer adds value beyond simply giving an adequate specification. Phase 2 items are listed in Sections 9A.8 and 15.4.

### 19.3 Final project position

The project should be presented as a **small, reproducible prompt-repair study with a working demonstration**. Its strength is the visible connection between an observed error, a constrained edit, and separately measured outcomes.

The final conclusion must answer what happened in the experiment, not repeat the working hypothesis as though it were already proven.

---

## 20. Sources

### Supplied PDFs

**[P1] Project compendium.** `Prompt_Projects_Times_New_Roman_Black_White(3).pdf`, page 65, Research Project 57 / PE-13, "PromptRepair: Automatic Repair Instead of Complete Prompt Rewriting." Source for the central research question, component-editing method, proposed comparators, original study scale, and metrics.

**[P2] Theory course plan.** `Prompt_Engineering_Course_Plan_Theory_and_Lab (1)(3).pdf`, page 2, Units 2-5; pages 5-6, teaching sessions on prompt debugging, evaluation, and capstone work. Source for the theory alignment, not for the proposed dataset or software architecture.

**[P3] Laboratory manual.** `Prompt_Engineering_Lab_Manual_Final_NN_Format(3).pdf`, pages 5-6 for Experiment 4, pages 6-7 for Experiment 5, pages 7-8 for Experiment 6, and pages 8-9 for Experiment 7. Supporting connections appear in the experiment list on page 2 and Experiments 1, 8, and 10 on pages 3, 9-10, and 11.

### Platform documentation: marimo molab (checked 19 October 2026)

None of these pages names the GPU model; the RTX 6000 detail comes from the student.

**[M1]** marimo blog, "molab, now with GPUs!" / "(Re-)introducing molab." Source for molab running on CoreWeave, notebooks running up to 12 hours, mirroring notebooks from GitHub, and sharing notebooks as apps or slides. `https://marimo.io/blog/reintroducing-molab`

**[M2]** marimo newsletter 26. Source for the public preview of molab on CoreWeave with GPU compute, described as free, and for reading and writing Google Drive files from molab notebooks. `https://marimo.io/blog/newsletter-26`

**[M3]** molab blog, "Announcing molab." Source for notebooks being public but undiscoverable by default, and for running a molab notebook locally with `uvx marimo edit <url>`. `https://molab.marimo.io/blog/announcing-molab`

**[M4]** marimo newsletter 22. Source for molab package sandboxing (packages recorded in the notebook file) and `marimo run` on a folder of notebooks. `https://marimo.io/blog/newsletter-22`

### Implementation documentation (checked 14 October 2026)

Model access, API details, and quotas may change; verify them again on implementation day one. [W1]-[W3] now matter only for the optional hosted-API arm (Section 14.9). [W4] is relevant to structured-output controls. [W5] (Streamlit) is retained for reference only, for anyone who uses a Streamlit front end instead of marimo.

**[W1] Google AI for Developers: Billing.** Source for the existence of a model-dependent Free Tier and the need to distinguish local execution from hosted-service data handling.

`https://ai.google.dev/gemini-api/docs/billing`

**[W2] Google AI for Developers: Rate limits.** Source for request/token limits, project-specific quota inspection, and variable capacity.

`https://ai.google.dev/gemini-api/docs/rate-limits`

**[W3] Google AI for Developers: Getting started.** Source for the official Google GenAI SDK and current API integration guidance.

`https://ai.google.dev/gemini-api/docs/get-started`

**[W4] Google AI for Developers: Structured outputs.** Source for structured response support and the distinction between syntactic validity and semantic correctness.

`https://ai.google.dev/gemini-api/docs/structured-output`

**[W5] Streamlit documentation: Install Streamlit using command line.** Source for local environment setup and running a Streamlit application.

`https://docs.streamlit.io/get-started/installation/command-line`

**[W6] Python 3.11 documentation: JSON encoder and decoder.** Source for parser behavior, `object_pairs_hook`, and `parse_constant`.

`https://docs.python.org/3.11/library/json.html`

### Published literature (found by web search on 19 October 2026)

Only abstracts and retrieved passages were read, not full texts. Verify authors, venues, and details before citing. Where an identifier is marked "from memory", it was not confirmed by the search.

**[L1]** Zhou, Y., Muresanu, A. I., Han, Z., Paster, K., Pitis, S., Chan, H., Ba, J. "Large Language Models Are Human-Level Prompt Engineers" (Automatic Prompt Engineer, APE). arXiv:2211.01910. `https://arxiv.org/pdf/2211.01910`

**[L2]** Yang, C., Wang, X., Lu, Y., Liu, H., Le, Q. V., Zhou, D., Chen, X. "Large Language Models as Optimizers" (OPRO). ICLR 2024. arXiv:2309.03409. `https://arxiv.org/pdf/2309.03409`

**[L3]** Pryzant, R., Iter, D., Li, J., Lee, Y. T., Zhu, C., Zeng, M. "Automatic Prompt Optimization with 'Gradient Descent' and Beam Search" (ProTeGi). EMNLP 2023. `https://aclanthology.org/2023.emnlp-main.494` ; arXiv:2305.03495.

**[L4]** Ye, Q., Ahmed, M., Pryzant, R., Khani, F. "Prompt Engineering a Prompt Engineer" (PE2). Findings of ACL 2024. arXiv:2311.05661. `https://arxiv.org/abs/2311.05661`

**[L5]** "Local Prompt Optimization" (LPO). NAACL 2025, short paper. arXiv:2504.20355. `https://arxiv.org/pdf/2504.20355` (Authors not retrieved; add from the PDF.)

**[L6]** Khattab, O., et al. "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." ICLR 2024. arXiv:2310.03714 (identifier from memory; verify).

**[L7]** Opsahl-Ong, K., et al. "Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs" (MIPRO / MIPROv2). arXiv:2406.11695. `https://arxiv.org/pdf/2406.11695`

**[L8]** Yuksekgonul, M., et al. "TextGrad: Automatic 'Differentiation' via Text." arXiv:2406.07496 (identifier from memory; verify).

**[L9]** Agrawal, L. A., et al. "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning." arXiv:2507.19457. `https://arxiv.org/abs/2507.19457` ; code: `https://github.com/gepa-ai/gepa`

**[L10]** "StraGo: Harnessing Strategic Guidance for Prompt Optimization." arXiv:2410.08601. `https://arxiv.org/pdf/2410.08601`

**[L11]** Chen, D., et al. "Learning from Prompt itself: the Hierarchical Attribution Prompt Optimization" (HAPO). arXiv:2601.02683. `https://arxiv.org/abs/2601.02683v1`

**[L12]** "Two-Stage Prompt Optimization for Few-Shot Relation Extraction: From Reasoning-Guided Search to Gradient-Guided Refinement" (introduces GradPO). arXiv:2606.29639. `https://arxiv.org/pdf/2606.29639`

**[L13]** Zhao, C., Wang, X., Zhao, P., Huang, Y., Lu, J., Liu, Z., Lin, Q., Rajmohan, S., Zhang, D. "Gradient-Guided Multi-Judge Prompt Optimization" (GMPO), 2026. Listing retrieved at `https://aclanthology.org/people/ziqian-liu-7155/` ; venue and identifier not retrieved.

**[L14]** "SkillOpt: Executive Strategy for Self-Evolving Agent Skills." arXiv:2605.23904. `https://arxiv.org/pdf/2605.23904`

**[L15]** "Contrastive Reflection for Iterative Prompt Optimization." arXiv:2606.30840. `https://arxiv.org/pdf/2606.30840`

**[L16]** Kong, W., Hombaiah, S. A., Zhang, M., Mei, Q., Bendersky, M. "PRewrite: Prompt Rewriting with Reinforcement Learning." arXiv:2401.08189. `https://arxiv.org/pdf/2401.08189`

**[L17]** "Prompts Are Programs Too! Understanding How Developers Build Software Containing Prompts." arXiv:2409.12447. `https://arxiv.org/pdf/2409.12447` (Authors not retrieved.)

**[L18]** "A Taxonomy of Prompt Defects in LLM Systems." arXiv:2509.14404. `https://arxiv.org/pdf/2509.14404`

---

**End of methodology guide.** The experimental design, 60-notice pilot, prompt templates, implementation layout, and timeline are proposed adaptations. They are not empirical results or additional requirements asserted by the supplied PDFs.
