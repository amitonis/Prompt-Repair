# Broader prompt repair: research and an 8–10-hour run plan

**Prepared:** 30 October 2026  
**Purpose:** choose a broader project direction and a defensible single-run experiment.  
**Current hardware evidence:** Molab previously exposed an NVIDIA RTX PRO 6000 Blackwell Server Edition with 94.97 GiB total GPU memory. The next runtime must be checked again; availability and speed can vary.

## Recommendation

Do not start an 8–10-hour weight-training run until a cross-task repair dataset has been prepared and checked. The current project has task examples, but it does not have a reliable set of examples mapping a defective prompt and its observed errors to a correct repaired prompt.

The project’s next research experiment should compare prompt-repair methods across several task families while keeping the executor model fixed. This is closer to the current research question, requires no model-weight training, and lets each task use its own checkable metric. After that establishes a useful repair-data pipeline, an 8–10-hour QLoRA adapter run is a reasonable next experiment.

If the requirement is specifically to change model weights in one run, use QLoRA/LoRA supervised fine-tuning of a 7B–8B repair model, with the training data, task splits, and validation scripts ready beforehand. The exact training duration cannot be promised from GPU memory alone; benchmark throughput first and project from measured tokens per second.

## What the research says

### Prompt optimization can generalize across task types

PromptWizard evaluates feedback-driven prompt evolution on 45 tasks spanning instruction induction, reasoning, and math. This supports a broader repair project, but its approach depends on task-specific feedback and evaluation; it is not a universal prompt editor that can be scored without task examples or a metric. [PromptWizard, Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.1025/)

GEPA evolves prompts using natural-language feedback from model traces and a task metric. Its paper reports results across six tasks and separates train feedback from validation-based candidate selection. It optimizes textual prompts while leaving the executor model weights frozen. The official DSPy guidance similarly recommends a separate held-out test set and warns that reusing training examples for selection can overfit. [GEPA paper](https://arxiv.org/abs/2507.19457), [DSPy GEPA set guidance](https://dspy.ai/current/getting-started/gepa-optimization/)

MIPRO also searches combinations of instructions and demonstrations. Its paper uses minibatch evaluation, a Bayesian surrogate, and repeated runs; the reported trial budgets vary by task. This suggests an 8–10-hour study should record candidate evaluations and token/runtime cost, not only wall-clock time. [MIPRO paper](https://aclanthology.org/2024.emnlp-main.525/)

### Prompt optimization does not reliably help every task

A 2026 study reports that 49% of 72 optimization runs on one executor model scored below the zero-shot baseline across four tasks; results also varied by task and model. The authors propose a low-cost headroom screen before investing in larger searches. That result argues for multiple task families and a baseline in the proposed project, rather than treating one successful prompt repair as evidence of general improvement. [Prompt Optimization Is a Coin Flip](https://arxiv.org/abs/2604.14585)

### Broad training requires diverse, balanced tasks

Super-NaturalInstructions contains 1,616 tasks across 76 task types and evaluates generalization to tasks held out from training. The FLAN Collection reports that task balancing and mixed prompt formats matter for instruction tuning. These works support splitting by task, not randomly mixing examples from every task into both train and test. [Super-NaturalInstructions](https://aclanthology.org/2022.emnlp-main.340/), [The FLAN Collection](https://proceedings.mlr.press/v202/longpre23a.html)

### QLoRA makes a single-GPU adapter run possible, but data quality is central

QLoRA trains low-rank adapters through a frozen 4-bit base model. The original paper demonstrated 65B-parameter fine-tuning on a single 48 GB GPU, including a 24-hour professional-GPU run; it does not establish a guaranteed time for this project’s model, dataset, or GPU. The authors also report that dataset quality mattered more than raw dataset size in one chatbot comparison. Hugging Face PEFT and TRL provide current QLoRA and supervised fine-tuning workflows. [QLoRA paper](https://arxiv.org/abs/2305.14314), [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization), [TRL SFT Trainer](https://huggingface.co/docs/trl/en/sft_trainer)

## What a broad repair system needs as training data

Raw task input/output pairs teach a model to do tasks; they do not directly teach it to repair prompts. A repair-training record should contain:

1. The task description and the original prompt.
2. A small set of example inputs and expected outputs.
3. The original prompt’s observed failures, preferably with structured error codes and concrete examples.
4. A repaired prompt that addresses those failures without hard-coding the examples.
5. A task-specific scorer that can compare the original and repaired prompts.

Build the initial dataset across at least six task families: extraction, classification, structured output, short-form QA, rewriting, and summarization. Use public, licensed task data where possible. Create prompt defects systematically (missing fields, wrong boundaries, conflicting rules, distractors, and formatting failures), then accept a repair target only if it improves on separate examples. Human-review a sample of generated repairs. Keep all examples from a task or dataset family within one split to measure transfer to genuinely unseen tasks.

## Proposed 8–10-hour QLoRA run

This is a plan for one reproducible experiment run, not a promise that training will finish in exactly eight hours. Prepare the dataset and install/pin packages before starting the clock. Keep a single run ID, but save resumable checkpoints and logs to persistent storage.

| Elapsed | Work | Decision/output |
| --- | --- | --- |
| 0:00–0:30 | Verify GPU, packages, dataset hashes, and split boundaries; record the untouched base-model score. | Abort early if memory or data checks fail. |
| 0:30–1:00 | Load the 7B–8B base model in 4-bit, attach LoRA, and measure a short training throughput sample. | Project steps/hour from measured tokens/sec; cap the run at 8–10 hours. |
| 1:00–7:00 | SFT the repair adapter on the prepared prompt-repair records. Save checkpoints periodically; validate after each fixed step interval. | Select checkpoint by task-family validation score, not training loss alone. |
| 7:00–8:00 | Compare base model versus adapter on the validation tasks using the same repair cases and metrics. | Freeze one selected checkpoint and config. |
| 8:00–9:00 | Evaluate once on held-out task families; produce per-family scores, paired differences, JSON/schema validity where applicable, and cost/time. | This is the final result; do not use this split to choose another checkpoint. |
| 9:00–10:00 | Save adapter, tokenizer/config, data manifest and hashes, logs, predictions, and a run report; allow time for export or recovery. | Complete reproducible artifact set. |

The single-run limit means results will be exploratory unless repeated with independent training seeds. Save checkpoints every 30–60 minutes and keep an early-stop rule for validation stagnation or regression. An 8–10-hour session without checkpoints is vulnerable to runtime interruption and loses useful work.

## Alternative: use the 8–10 hours for prompt optimization first

This is the recommended first long run for this project:

- Use a frozen executor model and compare the base prompt, the current repair method, and one established optimizer such as GEPA/MIPRO-style reflective search.
- Use 6–10 task families, each with train/feedback and validation examples; reserve whole task families or datasets for final transfer evaluation.
- Set a fixed rollout/token budget per method so the comparison is fair. Save every candidate and its metric, not only the winning prompt.
- Run a 10–20-candidate headroom screen first; continue on tasks where candidate prompts show measurable improvement over the baseline, and report flat tasks too.
- Reserve the final held-out evaluation for one pass after all choices are frozen.

This approach improves the repair process without claiming that a short adapter run has made the model broadly capable. It also creates scored repair trajectories that can later become training data for QLoRA.

## Decision

**Recommended order:** (1) implement a multi-task evaluator and repair-record format; (2) run the 8–10-hour fixed-model optimization study; (3) curate the resulting high-quality repair trajectories; (4) train and compare one QLoRA adapter as a follow-on. If the project specifically requires weight retraining now, do steps 1 and 3 before the long run; do not spend the GPU window generating and debugging the dataset from scratch.

## Sources reviewed

- Agrawal et al., [GEPA](https://arxiv.org/abs/2507.19457), 2025/2026.
- Agarwal et al., [PromptWizard](https://aclanthology.org/2025.findings-acl.1025/), ACL Findings 2025.
- Opsahl-Ong et al., [MIPRO](https://aclanthology.org/2024.emnlp-main.525/), EMNLP 2024.
- Zhang et al., [Prompt Optimization Is a Coin Flip](https://arxiv.org/abs/2604.14585), 2026 preprint.
- Wang et al., [Super-NaturalInstructions](https://aclanthology.org/2022.emnlp-main.340/), EMNLP 2022.
- Longpre et al., [The FLAN Collection](https://proceedings.mlr.press/v202/longpre23a.html), ICML 2023.
- Dettmers et al., [QLoRA](https://arxiv.org/abs/2305.14314), NeurIPS 2023.
- Official implementation guidance: [DSPy GEPA](https://dspy.ai/current/getting-started/gepa-optimization/), [Hugging Face PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization), and [TRL SFT Trainer](https://huggingface.co/docs/trl/en/sft_trainer).
