"""Summarize the single-setting D6 final-test artifact and paired contrasts."""

from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
INPUT = RUNS / "d6_final_test.json"
ORIGINAL_DEVELOPMENT_INPUT = RUNS / "original_development.json"
ORIGINAL_FINAL_TEST_INPUT = RUNS / "original_final_test.json"
RESTRICTED_DEVELOPMENT_INPUT = RUNS / "restricted_baseline_development.json"
RESTRICTED_FINAL_TEST_INPUT = RUNS / "restricted_baseline_final_test.json"
JSON_OUTPUT = RUNS / "d6_statistics.json"
MARKDOWN_OUTPUT = RUNS / "FINAL_REPORT.md"
FIELDS = ("event", "date", "venue")
SEED = 20261014
N_RESAMPLES = 10_000


def paired_cluster_bootstrap(
    differences: list[tuple[str, float]], *, seed: int = SEED
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for family, difference in differences:
        grouped.setdefault(family, []).append(float(difference))
    if not grouped:
        raise ValueError("No paired family data")
    clusters = list(grouped.values())
    total_notices = sum(map(len, clusters))
    observed = sum(sum(cluster) for cluster in clusters) / total_notices
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(N_RESAMPLES):
        sample = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        sample_n = sum(map(len, sample))
        draws.append(sum(sum(cluster) for cluster in sample) / sample_n)
    draws.sort()

    def percentile(q: float) -> float:
        position = q * (len(draws) - 1)
        lower = int(position)
        upper = min(lower + 1, len(draws) - 1)
        weight = position - lower
        return draws[lower] * (1 - weight) + draws[upper] * weight

    p_value = min(
        1.0,
        2 * min(
            sum(value <= 0 for value in draws) / N_RESAMPLES,
            sum(value >= 0 for value in draws) / N_RESAMPLES,
        ),
    )
    return {
        "observed_delta": observed,
        "ci_low": percentile(0.025),
        "ci_high": percentile(0.975),
        "bootstrap_p_descriptive": p_value,
        "bootstrap_tail_counts": {
            "resamples_at_or_below_zero": sum(value <= 0 for value in draws),
            "resamples_at_or_above_zero": sum(value >= 0 for value in draws),
        },
        "n_notices": total_notices,
        "n_families": len(clusters),
        "n_resamples": N_RESAMPLES,
        "seed": seed,
        "confidence": 0.95,
    }


def exact_sign_flip(
    differences: list[tuple[str, float]], *, scale: int = 1
) -> dict[str, Any]:
    totals: dict[str, int] = {}
    for family, difference in differences:
        scaled = difference * scale
        if not math.isclose(scaled, round(scaled), abs_tol=1e-9):
            raise ValueError("Scale does not make family differences integral")
        totals[family] = totals.get(family, 0) + round(scaled)
    values = list(totals.values())
    if not values:
        raise ValueError("No paired family data")
    observed = abs(sum(values))
    extreme = sum(
        abs(sum(sign * value for sign, value in zip(signs, values))) >= observed
        for signs in itertools.product((1, -1), repeat=len(values))
    )
    return {
        "p_descriptive": extreme / (2 ** len(values)),
        "n_families": len(values),
        "n_sign_patterns": 2 ** len(values),
        "family_total_scale": scale,
    }


def exact_mcnemar(a: dict[str, bool], b: dict[str, bool]) -> dict[str, Any]:
    a_only = sum(a[key] and not b[key] for key in a)
    b_only = sum(b[key] and not a[key] for key in a)
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = min(
            1.0,
            2
            * sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1))
            / (2**discordant),
        )
    return {
        "a_only": a_only,
        "b_only": b_only,
        "discordant_pairs": discordant,
        "exact_two_sided_p_sensitivity": p_value,
    }


def _arm_correctness(arm: dict[str, Any]) -> dict[str, bool]:
    return {row["id"]: bool(row["score"]["record_correct"]) for row in arm["records"]}


def _valid_field_counts(arm: dict[str, Any]) -> dict[str, int]:
    counts = dict.fromkeys(FIELDS, 0)
    for row in arm["records"]:
        score = row["score"]
        if score["json_valid"] and score["schema_valid"]:
            for field in FIELDS:
                counts[field] += bool(score["field_correct"][field])
    return counts


def build_report() -> dict[str, Any]:
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    arms = payload["results"]
    required = {"L-LLM", "rewrite", "oracle-section", "oracle"}
    if not required <= set(arms):
        raise ValueError(f"Final test artifact is missing arms: {sorted(required - set(arms))}")
    oracle_records = arms["oracle"]["records"]
    oracle_ids = [row["id"] for row in oracle_records]
    family_by_id = {row["id"]: row["family_id"] for row in oracle_records}
    if len(oracle_ids) != 40 or len(set(oracle_ids)) != 40:
        raise ValueError("Expected 40 unique final-test notices")

    arm_summaries: dict[str, Any] = {}
    correctness = {name: _arm_correctness(arm) for name, arm in arms.items()}
    for name, arm in arms.items():
        records = arm["records"]
        correct = sum(correctness[name].values())
        n = len(records)
        arm_summaries[name] = {
            "correct": correct,
            "attempted": n,
            "strict_accuracy": correct / n if n else None,
            "json_valid": sum(bool(row["score"]["json_valid"]) for row in records),
            "schema_valid": sum(bool(row["score"]["schema_valid"]) for row in records),
            "strict_field_correct": _valid_field_counts(arm),
        }

    p1 = [
        (family_by_id[notice_id], float(correctness["L-LLM"][notice_id]) - float(correctness["rewrite"][notice_id]))
        for notice_id in oracle_ids
    ]
    p1_stats = {
        "family_cluster_bootstrap": paired_cluster_bootstrap(p1),
        "exact_sign_flip": exact_sign_flip(p1),
        "mcnemar_sensitivity": exact_mcnemar(correctness["L-LLM"], correctness["rewrite"]),
    }

    random_names = [name for name in arms if name.startswith("random_")]
    if len(random_names) != 4:
        raise ValueError("Expected all four random-section candidates")
    random_rates = [arm_summaries[name]["strict_accuracy"] for name in random_names]
    random_mean = sum(random_rates) / 4
    p2 = []
    for notice_id in oracle_ids:
        random_expected = sum(float(correctness[name][notice_id]) for name in random_names) / 4
        difference = float(correctness["oracle-section"][notice_id]) - random_expected
        p2.append((family_by_id[notice_id], difference))
    p2_stats = {
        "oracle_section_accuracy": arm_summaries["oracle-section"]["strict_accuracy"],
        "random_section_mean_accuracy": random_mean,
        "observed_delta": arm_summaries["oracle-section"]["strict_accuracy"] - random_mean,
        "family_cluster_bootstrap": paired_cluster_bootstrap(p2),
        "exact_sign_flip": exact_sign_flip(p2, scale=4),
        "mcnemar_sensitivity": "Not applicable to the mean of four random-section prompts.",
    }

    oracle_without_punctuation = [
        row for row in oracle_records if row["id"] not in {"TEST-001", "TEST-002", "TEST-003", "TEST-004"}
    ]
    oracle_sensitivity_correct = sum(bool(row["score"]["record_correct"]) for row in oracle_without_punctuation)

    environment_path = RUNS / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8")) if environment_path.exists() else {}
    original_gate = None
    if ORIGINAL_DEVELOPMENT_INPUT.exists():
        gate_payload = json.loads(ORIGINAL_DEVELOPMENT_INPUT.read_text(encoding="utf-8"))
        original_gate = {
            "notices": gate_payload["notices"],
            "failures": gate_payload["failures"],
            "strict_accuracy": (gate_payload["notices"] - gate_payload["failures"]) / gate_payload["notices"],
            "include_original_prompt_arm": gate_payload["include_original_prompt_arm"],
            "json_valid": sum(bool(row["score"]["json_valid"]) for row in gate_payload["records"]),
            "elapsed_s": gate_payload.get("elapsed_s"),
            "failure_mode": "All 20 responses were wrapped in Markdown JSON code fences and failed strict JSON parsing.",
        }
    original_final = None
    if ORIGINAL_FINAL_TEST_INPUT.exists():
        original_final_payload = json.loads(
            ORIGINAL_FINAL_TEST_INPUT.read_text(encoding="utf-8")
        )
        original_final_arm = {"records": original_final_payload["records"]}
        original_final_correct = sum(
            bool(row["score"]["record_correct"])
            for row in original_final_payload["records"]
        )
        original_final_n = len(original_final_payload["records"])
        original_final = {
            "correct": original_final_correct,
            "attempted": original_final_n,
            "strict_accuracy": original_final_correct / original_final_n,
            "json_valid": sum(
                bool(row["score"]["json_valid"])
                for row in original_final_payload["records"]
            ),
            "schema_valid": sum(
                bool(row["score"]["schema_valid"])
                for row in original_final_payload["records"]
            ),
            "strict_field_correct": _valid_field_counts(original_final_arm),
            "elapsed_s": original_final_payload.get("elapsed_s"),
            "issue_counts": {
                issue: sum(
                    issue in row["score"]["issues"]
                    for row in original_final_payload["records"]
                )
                for issue in sorted({
                    issue
                    for row in original_final_payload["records"]
                    for issue in row["score"]["issues"]
                })
            },
        }
        arm_summaries["original"] = {
            key: value for key, value in original_final.items() if key != "elapsed_s"
        }
    restricted_development = None
    if RESTRICTED_DEVELOPMENT_INPUT.exists():
        restricted_dev_payload = json.loads(
            RESTRICTED_DEVELOPMENT_INPUT.read_text(encoding="utf-8")
        )
        restricted_development = {
            "correct": restricted_dev_payload["correct"],
            "attempted": restricted_dev_payload["notices"],
            "strict_accuracy": restricted_dev_payload["strict_accuracy"],
            "json_valid": restricted_dev_payload["json_valid"],
            "schema_valid": restricted_dev_payload["schema_valid"],
            "elapsed_s": restricted_dev_payload.get("elapsed_s"),
        }
    restricted_final = None
    if RESTRICTED_FINAL_TEST_INPUT.exists():
        restricted_final_payload = json.loads(
            RESTRICTED_FINAL_TEST_INPUT.read_text(encoding="utf-8")
        )
        restricted_final_records = restricted_final_payload["records"]
        restricted_final_correct = sum(
            bool(row["score"]["record_correct"]) for row in restricted_final_records
        )
        restricted_final_n = len(restricted_final_records)
        restricted_final_arm = {"records": restricted_final_records}
        restricted_final = {
            "correct": restricted_final_correct,
            "attempted": restricted_final_n,
            "strict_accuracy": restricted_final_correct / restricted_final_n,
            "json_valid": sum(
                bool(row["score"]["json_valid"]) for row in restricted_final_records
            ),
            "schema_valid": sum(
                bool(row["score"]["schema_valid"]) for row in restricted_final_records
            ),
            "strict_field_correct": _valid_field_counts(restricted_final_arm),
            "execution_note": restricted_final_payload.get("execution_note"),
        }
        arm_summaries["restricted_baseline"] = restricted_final
    expected_transformers = "5.20.2"
    observed_transformers = environment.get("serving_library_version")
    deviations = []
    if observed_transformers and observed_transformers != expected_transformers:
        deviations.append({
            "item": "Transformers version",
            "planned": expected_transformers,
            "observed": observed_transformers,
            "note": "Final run used the pasted Molab environment record.",
        })

    report = {
        "study": "PromptRepair single-model pilot",
        "setting": "D6",
        "model": environment.get("model_id", "Qwen/Qwen2.5-14B-Instruct"),
        "test_notices": 40,
        "test_families": len(set(family_by_id.values())),
        "elapsed_s": payload.get("elapsed_s"),
        "scope": "Descriptive single-setting results; only D6 survived screening.",
        "original_prompt_development_gate": original_gate,
        "original_prompt_final_test": original_final,
        "restricted_baseline_development": restricted_development,
        "restricted_baseline_final_test": restricted_final,
        "not_evaluated_preregistered_arms": ([
            {
                "arm": "original naturalistic prompt",
                "status": "development gate passed; final-test evaluation not run",
                "analysis_plan_rule": "include only if original prompt has at least 3 development failures",
            }
        ] if original_final is None else []) + [
            *([{
                "arm": "restricted baseline (constraints and output_format only)",
                "status": (
                    "development run complete; final-test evaluation not run"
                    if restricted_development
                    else "not run"
                ),
            }] if restricted_final is None else []),
            {"arm": "L-Probe", "status": "exploratory arm not run"},
        ],
        "arm_results": arm_summaries,
        "primary_comparisons": {
            "P1_L-LLM_vs_rewrite": p1_stats,
            "P2_oracle-section_vs_random-section-mean": p2_stats,
        },
        "sensitivity": {
            "oracle_excluding_TEST-001_to_TEST-004": {
                "correct": oracle_sensitivity_correct,
                "attempted": len(oracle_without_punctuation),
                "strict_accuracy": oracle_sensitivity_correct / len(oracle_without_punctuation),
            }
        },
        "inference_limit": (
            "Preregistered confirmatory across-setting tests and Holm correction are not run: "
            "one setting survived defect screening. Within-setting intervals and p-values are descriptive."
        ),
        "runtime_deviations": deviations,
        "random_section_candidates": random_names,
        "statistics_method": {
            "bootstrap": "10,000 paired family-cluster resamples, seed 20261014, percentile 95% CI",
            "sign_flip": "Exact enumeration over family totals; P2 scaled by four to preserve the mean-of-four contrast",
            "mcnemar": "Exact two-sided sensitivity analysis for P1 only",
        },
    }
    return report


def main() -> int:
    report = build_report()
    original_gate = report["original_prompt_development_gate"]
    original_final = report["original_prompt_final_test"]
    restricted_development = report["restricted_baseline_development"]
    restricted_final = report["restricted_baseline_final_test"]
    p1 = report["primary_comparisons"]["P1_L-LLM_vs_rewrite"]
    targeted_result = report["arm_results"]["L-LLM"]
    random_format_result = report["arm_results"]["random_output_format"]
    JSON_OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# PromptRepair final report: D6",
        "",
        f"- Model: {report['model']}",
        f"- Test notices: {report['test_notices']} across {report['test_families']} families",
        f"- Final evaluation time: {report['elapsed_s']:.2f} seconds",
        f"- Scope: {report['scope']}",
        "",
        "## Main finding",
        "",
        (
            f"Targeted repair and full rewrite both scored "
            f"{targeted_result['correct']}/{targeted_result['attempted']} "
            f"({targeted_result['strict_accuracy']:.1%}); their observed paired "
            f"difference was {p1['family_cluster_bootstrap']['observed_delta']:+.1%}. "
            "In this retained setting, targeted repair matched rewriting but did "
            "not outperform it. Only one setting survived screening, so this is "
            "descriptive evidence for this dataset, not a general result."
        ),
        (
            f"The output-format-only random-section arm also scored "
            f"{random_format_result['correct']}/{random_format_result['attempted']} "
            f"({random_format_result['strict_accuracy']:.1%}); "
            + (
                f"the constraints-plus-output-format baseline scored "
                f"{restricted_final['correct']}/{restricted_final['attempted']} "
                f"({restricted_final['strict_accuracy']:.1%}). "
                if restricted_final
                else "the restricted-baseline final test was not run. "
            )
            + "These matched scores make output-format instructions a relevant "
            "factor to investigate, but do not isolate a causal effect."
        ),
        "",
        "## Final test accuracy",
        "",
        "| Arm | Strict accuracy | JSON valid | Schema valid |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, result in report["arm_results"].items():
        lines.append(
            f"| {name} | {result['correct']}/{result['attempted']} ({result['strict_accuracy']:.1%}) "
            f"| {result['json_valid']}/{result['attempted']} | {result['schema_valid']}/{result['attempted']} |"
        )
    p2 = report["primary_comparisons"]["P2_oracle-section_vs_random-section-mean"]
    p1_boot = p1["family_cluster_bootstrap"]
    p2_boot = p2["family_cluster_bootstrap"]
    p2_bootstrap_display = (
        f"p<{1 / N_RESAMPLES:.4f} (0/{N_RESAMPLES} draws)"
        if p2_boot["bootstrap_p_descriptive"] == 0
        else f"{p2_boot['bootstrap_p_descriptive']:.4f}"
    )
    lines.extend([
        "",
        "## Paired comparisons (descriptive for this single retained setting)",
        "",
        f"- P1, L-LLM minus rewrite: {p1_boot['observed_delta']:+.1%}; 95% family-bootstrap CI "
        f"[{p1_boot['ci_low']:+.1%}, {p1_boot['ci_high']:+.1%}]; bootstrap p={p1_boot['bootstrap_p_descriptive']:.4f}; "
        f"exact family sign-flip p={p1['exact_sign_flip']['p_descriptive']:.4f}.",
        f"- P2, oracle-section minus random-section mean: {p2['observed_delta']:+.1%}; 95% family-bootstrap CI "
        f"[{p2_boot['ci_low']:+.1%}, {p2_boot['ci_high']:+.1%}]; bootstrap {p2_bootstrap_display}; "
        f"exact family sign-flip p={p2['exact_sign_flip']['p_descriptive']:.4f}.",
        "",
        "## Sensitivity and limits",
        "",
        f"Oracle accuracy excluding TEST-001–TEST-004: "
        f"{report['sensitivity']['oracle_excluding_TEST-001_to_TEST-004']['correct']}/36 "
        f"({report['sensitivity']['oracle_excluding_TEST-001_to_TEST-004']['strict_accuracy']:.1%}).",
        report["inference_limit"],
        (
            f"Original-prompt development gate: {original_gate['failures']}/{original_gate['notices']} "
            f"strict failures in {original_gate['elapsed_s']:.2f} seconds; the arm qualifies for inclusion. "
            f"All {original_gate['notices'] - original_gate['json_valid']} failures were malformed JSON "
            "because outputs were wrapped in Markdown code fences."
            + (
                f" Original-prompt final-test result: {original_final['correct']}/"
                f"{original_final['attempted']} ({original_final['strict_accuracy']:.1%}); "
                f"issue counts: {original_final['issue_counts']}."
                if original_final
                else " The original arm has not been run on the final test set."
            )
            if original_gate
            else "The original-prompt development inclusion gate remains unchecked."
        ),
        (
            f"Restricted-baseline development result: {restricted_development['correct']}/"
            f"{restricted_development['attempted']} ({restricted_development['strict_accuracy']:.1%}); "
            + (
                "final-test evaluation is pending."
                if not restricted_final
                else f"final-test result: {restricted_final['correct']}/"
                f"{restricted_final['attempted']} ({restricted_final['strict_accuracy']:.1%}); "
                f"{restricted_final['json_valid']}/{restricted_final['attempted']} valid JSON and "
                f"{restricted_final['schema_valid']}/{restricted_final['attempted']} valid schema. "
                "The run occurred after other final-test results were observed; no prompt changes were made."
            )
            if restricted_development
            else (
                f"Restricted-baseline final-test result: {restricted_final['correct']}/"
                f"{restricted_final['attempted']} ({restricted_final['strict_accuracy']:.1%})."
                if restricted_final
                else "The restricted baseline was not evaluated."
            )
        ),
        "## Failure analysis",
        "",
        "See [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md) for category-level accuracy, "
        "the five remaining error cases, and the possible punctuation-boundary "
        "ambiguity in three frozen labels.",
        "",
        "Exploratory L-Probe was not evaluated.",
    ])
    if report["runtime_deviations"]:
        lines.extend(["", "## Runtime deviation", ""])
        for deviation in report["runtime_deviations"]:
            lines.append(
                f"- {deviation['item']}: planned {deviation['planned']}; observed {deviation['observed']}."
            )
    MARKDOWN_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {JSON_OUTPUT}")
    print(f"Wrote {MARKDOWN_OUTPUT}")
    print(json.dumps(report["primary_comparisons"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
