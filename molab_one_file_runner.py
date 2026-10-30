# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "pytest",
#   "torch",
#   "transformers",
#   "accelerate",
#   "huggingface_hub",
# ]
# ///

"""Single-file Molab runner for PromptRepair.

Upload ``promptrepair_molab_bundle.zip`` through the first UI cell whenever a
Molab session resets. The notebook recreates the project, validates the data,
runs the tests, records the runtime, and can run a real-model development
smoke/baseline evaluation.

This project does not fine-tune a model. Its intervention is prompt repair, so
the model is loaded for deterministic inference and evaluation.
"""

from __future__ import annotations

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


# Cell 1: Import Marimo for notebook UI components.
@app.cell
def _():
    import marimo as mo

    return (mo,)


# Cell 2: Explain the runner and the order of operations.
@app.cell
def _(mo):
    mo.md(
        """
        # PromptRepair — one-file Molab runner

        This notebook is designed for a fresh Molab session. Upload the
        current `promptrepair_molab_bundle.zip` below, then run the cells in
        order.

        It will:

        1. recreate the project files;
        2. validate the 60-notice dataset;
        3. run the project's automated checks;
        4. record the GPU/runtime environment;
        5. optionally load Qwen2.5-14B and run a development evaluation.

        **There is no fine-tuning in this study.** The research question is
        about prompt repair, so the model is evaluated with fixed weights.
        """
    )
    return


# Cell 3: Show the ZIP file upload control.
@app.cell
def _(mo):
    bundle_upload = mo.ui.file(
        filetypes=[".zip"],
        multiple=False,
        kind="area",
        label="Upload promptrepair_molab_bundle.zip",
    )
    bundle_upload
    return (bundle_upload,)


# Cell 4: Extract the uploaded bundle and restore saved run artifacts.
@app.cell
def _(bundle_upload, mo):
    import hashlib as bundle_hashlib
    import json as bundle_json
    import shutil as bundle_shutil
    import sys as bundle_sys
    import zipfile as bundle_zipfile
    from pathlib import Path as BundlePath

    bundle_runtime_root = BundlePath("promptrepair_runtime").resolve()
    bundle_status = "Waiting for promptrepair_molab_bundle.zip upload."

    if bundle_upload.value:
        bundle_name = bundle_upload.name() or "uploaded_bundle.zip"
        bundle_bytes = bundle_upload.contents()
        if bundle_bytes is None:
            raise ValueError("The uploaded ZIP has no readable contents.")
        if not bundle_name.lower().endswith(".zip"):
            raise ValueError("Upload the promptrepair_molab_bundle.zip file.")

        if bundle_runtime_root.exists():
            bundle_shutil.rmtree(bundle_runtime_root)
        bundle_runtime_root.mkdir(parents=True, exist_ok=True)

        with bundle_zipfile.ZipFile(__import__("io").BytesIO(bundle_bytes)) as archive:
            for member in archive.infolist():
                member_path = (bundle_runtime_root / member.filename).resolve()
                if bundle_runtime_root not in member_path.parents:
                    raise ValueError(f"Unsafe ZIP member path: {member.filename}")
            archive.extractall(bundle_runtime_root)

        if str(bundle_runtime_root) not in bundle_sys.path:
            bundle_sys.path.insert(0, str(bundle_runtime_root))
        bundle_digest = bundle_hashlib.sha256(bundle_bytes).hexdigest()
        bundle_metadata = {
            "uploaded_name": bundle_name,
            "bundle_sha256": bundle_digest,
            "runtime_root": str(bundle_runtime_root),
        }
        (bundle_runtime_root / "runs").mkdir(exist_ok=True)
        (bundle_runtime_root / "runs" / "bundle_metadata.json").write_text(
            bundle_json.dumps(bundle_metadata, indent=2) + "\n", encoding="utf-8"
        )
        bundle_status = f"Extracted {bundle_name} into {bundle_runtime_root}."

    mo.md(bundle_status)
    return bundle_runtime_root, bundle_status


# Cell 5: Validate the frozen dataset.
@app.cell
def _(bundle_runtime_root, mo):
    import contextlib as validation_contextlib
    import io as validation_io
    import runpy as validation_runpy

    validation_output = ""
    validation_ok = False
    if (bundle_runtime_root / "scripts" / "validate_data.py").exists():
        validation_buffer = validation_io.StringIO()
        with validation_contextlib.redirect_stdout(validation_buffer):
            try:
                validation_runpy.run_path(
                    str(bundle_runtime_root / "scripts" / "validate_data.py"),
                    run_name="__main__",
                )
            except SystemExit as validation_exit:
                if validation_exit.code not in (0, None):
                    raise
        validation_output = validation_buffer.getvalue()
        validation_ok = "PASSED: all dataset checks" in validation_output
    else:
        validation_output = "Upload the ZIP first, then rerun this cell."

    mo.md(
        "### Dataset validation\n\n`````text\n"
        + validation_output
        + "\n`````"
        + ("\n\n**Dataset validation passed.**" if validation_ok else "")
    )
    return validation_ok, validation_output


# Cell 6: Run the automated project checks.
@app.cell
def _(bundle_runtime_root, mo):
    import pytest as project_pytest

    tests_output = ""
    tests_exit_code = None
    if (bundle_runtime_root / "tests").exists():
        tests_result = project_pytest.main(
            [
                "-q",
                str(bundle_runtime_root / "tests"),
                "--basetemp=" + str(bundle_runtime_root / "pytest_tmp"),
            ]
        )
        tests_exit_code = int(tests_result)
        tests_output = f"pytest exit code: {tests_exit_code}"
    else:
        tests_output = "Upload the ZIP first, then rerun this cell."

    mo.md("### Automated tests\n\n" + tests_output)
    return tests_exit_code, tests_output


# Cell 7: Record the Molab Python, PyTorch, Transformers, CUDA, and GPU details.
@app.cell
def _(bundle_runtime_root, mo):
    import json as environment_json
    import platform as environment_platform
    import subprocess as environment_subprocess

    environment_record = {}
    try:
        import torch as environment_torch
        import transformers as environment_transformers

        environment_cuda = getattr(environment_torch, "cuda", None)
        environment_has_gpu = bool(environment_cuda and environment_cuda.is_available())
        environment_record = {
            "model_id": "Qwen/Qwen2.5-14B-Instruct",
            "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
            "serving_library": "transformers",
            "serving_library_version": getattr(environment_transformers, "__version__", None),
            "torch_version": getattr(environment_torch, "__version__", None),
            "dtype": "bfloat16",
            "quantization": "none",
            "os": environment_platform.platform(),
            "gpu_available": environment_has_gpu,
            "gpu_name": (
                environment_cuda.get_device_name(0) if environment_has_gpu else None
            ),
            "cuda_version": getattr(
                getattr(environment_torch, "version", None), "cuda", None
            ),
            "driver_version": (
                environment_subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip()
                or None
            ),
            "batch_size": 16,
            "recorded_by": "molab_one_file_runner.py",
        }
        environment_path = bundle_runtime_root / "runs" / "environment_live.json"
        environment_path.parent.mkdir(parents=True, exist_ok=True)
        environment_path.write_text(
            environment_json.dumps(environment_record, indent=2) + "\n",
            encoding="utf-8",
        )
        environment_text = environment_json.dumps(environment_record, indent=2)
    except ImportError as environment_error:
        environment_text = f"Runtime packages unavailable: {environment_error}"

    mo.md("### Live environment\n\n```json\n" + environment_text + "\n```")
    return environment_record, environment_text


# Cell 8: Show the optional three-notice model smoke-test button.
@app.cell
def _(mo):
    model_smoke_button = mo.ui.run_button(label="Load model and run 3-notice smoke test")
    model_smoke_button
    return (model_smoke_button,)


# Cell 9: Optionally load the model and run the smoke test.
@app.cell
def _(bundle_runtime_root, model_smoke_button, mo):
    import json as smoke_json
    import time as smoke_time

    smoke_text = "Click the smoke-test button after dataset validation and tests pass."
    if model_smoke_button.value:
        from src.defects import load_prompt as smoke_load_prompt
        from src.evaluator import score_response as smoke_score_response
        from src.prompt_renderer import render_prompt as smoke_render_prompt
        from src.storage import load_subset as smoke_load_subset
        from src.inference import LocalTransformersAdapter as SmokeAdapter

        smoke_config = smoke_json.loads(
            (bundle_runtime_root / "config" / "experiment.json").read_text(
                encoding="utf-8"
            )
        )
        smoke_model_config = smoke_config["models"]["extractor"]
        smoke_environment_config = smoke_config["environment"]
        smoke_adapter = SmokeAdapter(
            model_id=smoke_model_config["hf_model_id"],
            revision=smoke_model_config["revision"],
            dtype=smoke_model_config["dtype"],
            quantization=smoke_model_config["quantization"],
            batch_size=smoke_environment_config["batch_size"],
            max_new_tokens=smoke_model_config["max_new_tokens"],
            device="cuda",
        )
        smoke_cases = smoke_load_subset("feedback")[:3]
        smoke_sections = smoke_load_prompt("oracle")
        smoke_prompts = [
            smoke_render_prompt(smoke_sections, case["notice"]) for case in smoke_cases
        ]
        smoke_started = smoke_time.perf_counter()
        smoke_records = smoke_adapter.generate_batch(
            smoke_prompts,
            run_id="molab-smoke",
            seed=20261014,
        )
        smoke_elapsed = smoke_time.perf_counter() - smoke_started
        smoke_scores = [
            smoke_score_response(record.raw, case["gold"])
            for record, case in zip(smoke_records, smoke_cases)
        ]
        smoke_text = smoke_json.dumps(
            {
                "notices": len(smoke_cases),
                "elapsed_s": smoke_elapsed,
                "record_correct": [score["record_correct"] for score in smoke_scores],
                "raw_outputs": [record.raw for record in smoke_records],
            },
            indent=2,
        )
        (bundle_runtime_root / "runs" / "smoke_results.json").write_text(
            smoke_text + "\n", encoding="utf-8"
        )

    mo.md("### Model smoke test\n\n```json\n" + smoke_text + "\n```")
    return smoke_text


# Cell 10: Show the oracle development-baseline button.
@app.cell
def _(mo):
    baseline_button = mo.ui.run_button(
        label="Run oracle development baseline on 20 notices"
    )
    mo.md(
        "This is the first real experiment step. It uses only the feedback and "
        "selection notices, not the final test set. It may take several minutes.\n\n"
        + baseline_button.text
    )
    return (baseline_button,)


# Cell 11: Optionally evaluate the oracle prompt on all 20 development notices.
@app.cell
def _(baseline_button, bundle_runtime_root, mo):
    import json as baseline_json
    import time as baseline_time

    baseline_text = "Click the baseline button after the smoke test succeeds."
    if baseline_button.value:
        from src.defects import load_prompt as baseline_load_prompt
        from src.evaluator import score_response as baseline_score_response
        from src.prompt_renderer import render_prompt as baseline_render_prompt
        from src.storage import load_subset as baseline_load_subset
        from src.inference import LocalTransformersAdapter as BaselineAdapter

        baseline_config = baseline_json.loads(
            (bundle_runtime_root / "config" / "experiment.json").read_text(
                encoding="utf-8"
            )
        )
        baseline_model = baseline_config["models"]["extractor"]
        baseline_environment = baseline_config["environment"]
        baseline_adapter = BaselineAdapter(
            model_id=baseline_model["hf_model_id"],
            revision=baseline_model["revision"],
            dtype=baseline_model["dtype"],
            quantization=baseline_model["quantization"],
            batch_size=baseline_environment["batch_size"],
            max_new_tokens=baseline_model["max_new_tokens"],
            device="cuda",
        )
        baseline_cases = (
            baseline_load_subset("feedback") + baseline_load_subset("selection")
        )
        baseline_sections = baseline_load_prompt("oracle")
        baseline_prompts = [
            baseline_render_prompt(baseline_sections, case["notice"])
            for case in baseline_cases
        ]
        baseline_started = baseline_time.perf_counter()
        baseline_records = baseline_adapter.generate_batch(
            baseline_prompts,
            run_id="oracle-development",
            seed=20261014,
        )
        baseline_elapsed = baseline_time.perf_counter() - baseline_started
        baseline_rows = []
        for case, record in zip(baseline_cases, baseline_records):
            score = baseline_score_response(record.raw, case["gold"])
            baseline_rows.append(
                {
                    "id": case["id"],
                    "split": case["split"],
                    "category": case["category"],
                    "raw": record.raw,
                    "status": record.status,
                    "score": score,
                }
            )
        baseline_accuracy = sum(row["score"]["record_correct"] for row in baseline_rows) / len(
            baseline_rows
        )
        baseline_payload = {
            "arm": "oracle",
            "notices": len(baseline_rows),
            "strict_accuracy": baseline_accuracy,
            "elapsed_s": baseline_elapsed,
            "records": baseline_rows,
        }
        baseline_text = baseline_json.dumps(baseline_payload, indent=2)
        (bundle_runtime_root / "runs" / "oracle_development.json").write_text(
            baseline_text + "\n", encoding="utf-8"
        )

    mo.md("### Oracle development baseline\n\n```json\n" + baseline_text + "\n```")
    return baseline_text


# Cell 12: Show the optional defect-screening button.
@app.cell
def _(mo):
    defect_screen_button = mo.ui.run_button(
        label="Run all eight defect-screening evaluations"
    )
    mo.md(
        "This runs the eight injected defects on the same 20 development "
        "notices and applies the preregistered screening rules. It may take "
        "a few minutes.\n\n"
        + defect_screen_button.text
    )
    return (defect_screen_button,)


# Cell 13: Optionally evaluate all eight defects on development data.
@app.cell
def _(bundle_runtime_root, defect_screen_button, mo):
    import json as screening_json
    import time as screening_time

    screening_text = "Click the defect-screening button after the 90% oracle baseline passes."
    if defect_screen_button.value:
        from src.defects import inject_defect as screening_inject_defect
        from src.defects import load_defects as screening_load_defects
        from src.defects import load_prompt as screening_load_prompt
        from src.evaluator import score_response as screening_score_response
        from src.experiment_runner import screen_defects as screening_screen_defects
        from src.prompt_renderer import render_prompt as screening_render_prompt
        from src.storage import load_subset as screening_load_subset
        from src.inference import LocalTransformersAdapter as ScreeningAdapter

        screening_config = screening_json.loads(
            (bundle_runtime_root / "config" / "experiment.json").read_text(
                encoding="utf-8"
            )
        )
        screening_model = screening_config["models"]["extractor"]
        screening_environment = screening_config["environment"]
        screening_adapter = ScreeningAdapter(
            model_id=screening_model["hf_model_id"],
            revision=screening_model["revision"],
            dtype=screening_model["dtype"],
            quantization=screening_model["quantization"],
            batch_size=screening_environment["batch_size"],
            max_new_tokens=screening_model["max_new_tokens"],
            device="cuda",
        )
        screening_cases = (
            screening_load_subset("feedback") + screening_load_subset("selection")
        )
        screening_oracle_payload = screening_json.loads(
            (bundle_runtime_root / "runs" / "oracle_development.json").read_text(
                encoding="utf-8"
            )
        )
        screening_oracle_correct = {
            row["id"]: bool(row["score"]["record_correct"])
            for row in screening_oracle_payload["records"]
        }
        screening_sections = screening_load_prompt("oracle")
        screening_defects = screening_load_defects()
        screening_jobs = []
        screening_targets = {}
        screening_target_categories = {
            "D1": {"venue_absent_or_pending", "date_absent", "unresolved_conflict"},
            "D2": {"deadline_distractor"},
            "D3": {"explicit_correction"},
            "D4": {"venue_absent_or_pending", "date_absent", "unresolved_conflict"},
        }
        for screening_defect in screening_defects:
            screening_id = screening_defect["id"]
            screening_targets[screening_id] = [
                case["id"]
                for case in screening_cases
                if case["category"] in screening_target_categories.get(
                    screening_id, {case["category"]}
                )
            ]
            screening_sections_for_defect = screening_inject_defect(
                screening_sections, screening_defect
            )
            for case in screening_cases:
                screening_jobs.append(
                    (
                        screening_id,
                        case,
                        screening_render_prompt(
                            screening_sections_for_defect, case["notice"]
                        ),
                    )
                )

        screening_started = screening_time.perf_counter()
        screening_records = screening_adapter.generate_batch(
            [job[2] for job in screening_jobs],
            run_id="defect-screening",
            seed=20261014,
        )
        screening_elapsed = screening_time.perf_counter() - screening_started
        screening_correct = {defect["id"]: {} for defect in screening_defects}
        screening_rows = []
        for (screening_id, case, _), record in zip(screening_jobs, screening_records):
            screening_score = screening_score_response(record.raw, case["gold"])
            screening_correct[screening_id][case["id"]] = bool(
                screening_score["record_correct"]
            )
            screening_rows.append(
                {
                    "defect_id": screening_id,
                    "id": case["id"],
                    "raw": record.raw,
                    "status": record.status,
                    "score": screening_score,
                }
            )
        screening_results = screening_screen_defects(
            screening_oracle_correct,
            screening_correct,
            target_notices=screening_targets,
        )
        screening_payload = {
            "arm": "defect_screening",
            "notices_per_defect": len(screening_cases),
            "elapsed_s": screening_elapsed,
            "results": [result.__dict__ for result in screening_results],
            "records": screening_rows,
        }
        screening_text = screening_json.dumps(screening_payload, indent=2)
        (bundle_runtime_root / "runs" / "defect_screening.json").write_text(
            screening_text + "\n", encoding="utf-8"
        )

    mo.md("### Defect screening\n\n```json\n" + screening_text + "\n```")
    return screening_text


# Cell 14: Explain how to resume after a Molab session reset.
@app.cell
def _(bundle_runtime_root, mo):
    mo.md(
        """
        ## Session recovery

        Molab may reset the runtime. On the next session:

        1. open the updated `molab_one_file_runner.py`;
        2. upload the rebuilt `promptrepair_molab_bundle.zip`;
        3. run setup and validation cells, then run the original-prompt
           headroom gate when ready.

        The bundle includes the current source, frozen data, and saved run
        JSON/report artifacts, so earlier results are restored on upload. It
        does not rerun the locked final test. New results are still local to
        the Molab session; save or sync them before the session ends.

        If GPU memory is already occupied, restart the Molab runtime before
        loading the model. The runner checks free memory and stops with a
        recovery message before attempting a model load when space is low.
        """
    )
    return


# Cell 15: Show the original-prompt headroom-gate button.
@app.cell
def _(mo):
    original_gate_button = mo.ui.run_button(
        label="Run original-prompt development headroom gate"
    )
    mo.md(
        "Run this after setup and dataset validation. It evaluates the original "
        "prompt on feedback + selection only (20 development notices). The "
        "pre-registered inclusion gate is **at least 3 failures**. The final "
        "test set is not used.\n\n"
        + original_gate_button.text
    )
    return (original_gate_button,)


# Cell 16: Run and save the original-prompt headroom evaluation.
@app.cell
def _(bundle_runtime_root, mo, original_gate_button):
    import gc as original_gate_gc
    import json as original_gate_json
    import time as original_gate_time

    original_gate_text = "Click the development headroom gate button to run."
    if original_gate_button.value:
        import torch as original_gate_torch

        from src.defects import load_prompt as original_gate_load_prompt
        from src.evaluator import score_response as original_gate_score
        from src.inference import LocalTransformersAdapter as OriginalGateAdapter
        from src.prompt_renderer import render_prompt as original_gate_render
        from src.storage import load_subset as original_gate_load_subset

        original_gate_config = original_gate_json.loads(
            (bundle_runtime_root / "config" / "experiment.json").read_text(
                encoding="utf-8"
            )
        )
        original_gate_model = original_gate_config["models"]["extractor"]
        original_gate_env = original_gate_config["environment"]
        original_gate_cases = (
            original_gate_load_subset("feedback")
            + original_gate_load_subset("selection")
        )
        original_gate_sections = original_gate_load_prompt("original")
        original_gate_prompts = [
            original_gate_render(original_gate_sections, case["notice"])
            for case in original_gate_cases
        ]

        # Reclaim only unreachable allocations. A live model from an earlier
        # notebook cell cannot be freed here, so fail early and request restart.
        original_gate_gc.collect()
        if original_gate_torch.cuda.is_available():
            original_gate_torch.cuda.empty_cache()
            original_gate_free, original_gate_total = (
                original_gate_torch.cuda.mem_get_info()
            )
            original_gate_min_free = 32 * 1024**3
            if original_gate_free < original_gate_min_free:
                raise RuntimeError(
                    "Not enough free GPU memory to load the model safely: "
                    f"{original_gate_free / 1024**3:.1f} GiB free of "
                    f"{original_gate_total / 1024**3:.1f} GiB. Restart the "
                    "Molab runtime to release models retained by earlier cells, "
                    "then rerun this gate."
                )

        original_gate_adapter = OriginalGateAdapter(
            model_id=original_gate_model["hf_model_id"],
            revision=original_gate_model["revision"],
            dtype=original_gate_model["dtype"],
            quantization=original_gate_model["quantization"],
            batch_size=original_gate_env["batch_size"],
            max_new_tokens=original_gate_model["max_new_tokens"],
            device="cuda",
        )
        original_gate_started = original_gate_time.perf_counter()
        original_gate_generations = original_gate_adapter.generate_batch(
            original_gate_prompts,
            run_id="original-development-headroom",
            seed=20261014,
        )
        original_gate_elapsed = original_gate_time.perf_counter() - original_gate_started

        original_gate_rows = []
        for original_gate_case, original_gate_generation in zip(
            original_gate_cases, original_gate_generations
        ):
            original_gate_score_result = original_gate_score(
                original_gate_generation.raw, original_gate_case["gold"]
            )
            original_gate_rows.append(
                {
                    "id": original_gate_case["id"],
                    "split": original_gate_case["split"],
                    "category": original_gate_case["category"],
                    "raw": original_gate_generation.raw,
                    "status": original_gate_generation.status,
                    "score": original_gate_score_result,
                }
            )

        original_gate_failures = sum(
            not row["score"]["record_correct"] for row in original_gate_rows
        )
        original_gate_payload = {
            "arm": "original_prompt_headroom_gate",
            "notices": len(original_gate_rows),
            "failures": original_gate_failures,
            "include_original_prompt_arm": original_gate_failures >= 3,
            "gate_rule": "Include only if at least 3 development failures.",
            "elapsed_s": original_gate_elapsed,
            "records": original_gate_rows,
        }
        original_gate_path = (
            bundle_runtime_root / "runs" / "original_development.json"
        )
        original_gate_path.write_text(
            original_gate_json.dumps(original_gate_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        original_gate_text = original_gate_json.dumps(original_gate_payload, indent=2)

    mo.md(
        "### Original-prompt headroom result\n\n```json\n"
        + original_gate_text
        + "\n```"
    )
    return (original_gate_text,)


# Cell 17: Show the original-prompt final-test button.
@app.cell
def _(mo):
    original_test_button = mo.ui.run_button(
        label="Run original-prompt final test (40 notices)"
    )
    mo.md(
        "The development gate passed, so this evaluates the frozen original "
        "prompt once on all 40 test notices. It checks that the saved output "
        "does not already exist and scores only after generation is complete.\n\n"
        + original_test_button.text
    )
    return (original_test_button,)


# Cell 18: Evaluate and save the original-prompt final-test arm.
@app.cell
def _(bundle_runtime_root, mo, original_test_button):
    import gc as original_test_gc
    import json as original_test_json
    import time as original_test_time

    original_test_text = "Click the original-prompt final-test button to run."
    if original_test_button.value:
        import torch as original_test_torch

        from src.defects import load_prompt as original_test_load_prompt
        from src.evaluator import score_response as original_test_score
        from src.inference import LocalTransformersAdapter as OriginalTestAdapter
        from src.prompt_renderer import render_prompt as original_test_render
        from src.storage import load_subset as original_test_load_subset

        original_test_gate_path = (
            bundle_runtime_root / "runs" / "original_development.json"
        )
        if not original_test_gate_path.exists():
            raise FileNotFoundError(
                "Missing the saved original-prompt development gate result."
            )
        original_test_gate = original_test_json.loads(
            original_test_gate_path.read_text(encoding="utf-8")
        )
        if not original_test_gate.get("include_original_prompt_arm", False):
            raise ValueError(
                "The original prompt did not pass the preregistered inclusion gate."
            )

        original_test_path = (
            bundle_runtime_root / "runs" / "original_final_test.json"
        )
        if original_test_path.exists():
            raise FileExistsError(
                "original_final_test.json already exists; refusing to overwrite it."
            )

        original_test_config = original_test_json.loads(
            (bundle_runtime_root / "config" / "experiment.json").read_text(
                encoding="utf-8"
            )
        )
        original_test_model = original_test_config["models"]["extractor"]
        original_test_env = original_test_config["environment"]
        original_test_cases = original_test_load_subset("test", with_labels=False)
        original_test_sections = original_test_load_prompt("original")
        original_test_prompts = [
            original_test_render(original_test_sections, case["notice"])
            for case in original_test_cases
        ]

        original_test_gc.collect()
        if original_test_torch.cuda.is_available():
            original_test_torch.cuda.empty_cache()
            original_test_free, original_test_total = (
                original_test_torch.cuda.mem_get_info()
            )
            original_test_min_free = 32 * 1024**3
            if original_test_free < original_test_min_free:
                raise RuntimeError(
                    "Not enough free GPU memory to load the model safely: "
                    f"{original_test_free / 1024**3:.1f} GiB free of "
                    f"{original_test_total / 1024**3:.1f} GiB. Restart the "
                    "Molab runtime before running this arm."
                )

        original_test_adapter = OriginalTestAdapter(
            model_id=original_test_model["hf_model_id"],
            revision=original_test_model["revision"],
            dtype=original_test_model["dtype"],
            quantization=original_test_model["quantization"],
            batch_size=original_test_env["batch_size"],
            max_new_tokens=original_test_model["max_new_tokens"],
            device="cuda",
        )
        original_test_started = original_test_time.perf_counter()
        original_test_generations = original_test_adapter.generate_batch(
            original_test_prompts,
            run_id="original-final-test",
            seed=20261014,
        )
        original_test_elapsed = (
            original_test_time.perf_counter() - original_test_started
        )

        # Labels are loaded only after all test generations have completed.
        original_test_labeled_cases = original_test_load_subset("test")
        original_test_rows = []
        for original_test_case, original_test_generation in zip(
            original_test_labeled_cases, original_test_generations
        ):
            original_test_score_result = original_test_score(
                original_test_generation.raw, original_test_case["gold"]
            )
            original_test_rows.append(
                {
                    "id": original_test_case["id"],
                    "family_id": original_test_case["family_id"],
                    "split": original_test_case["split"],
                    "category": original_test_case["category"],
                    "raw": original_test_generation.raw,
                    "status": original_test_generation.status,
                    "score": original_test_score_result,
                }
            )

        original_test_correct = sum(
            bool(row["score"]["record_correct"]) for row in original_test_rows
        )
        original_test_payload = {
            "arm": "original",
            "setting_id": "naturalistic-original-prompt",
            "notices": len(original_test_rows),
            "correct": original_test_correct,
            "strict_accuracy": original_test_correct / len(original_test_rows),
            "json_valid": sum(
                bool(row["score"]["json_valid"]) for row in original_test_rows
            ),
            "schema_valid": sum(
                bool(row["score"]["schema_valid"]) for row in original_test_rows
            ),
            "elapsed_s": original_test_elapsed,
            "generation_seed": 20261014,
            "records": original_test_rows,
        }
        original_test_path.write_text(
            original_test_json.dumps(original_test_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        original_test_text = original_test_json.dumps(
            {
                key: value
                for key, value in original_test_payload.items()
                if key != "records"
            },
            indent=2,
        )

    mo.md(
        "### Original-prompt final-test result\n\n```json\n"
        + original_test_text
        + "\n```"
    )
    return (original_test_text,)


if __name__ == "__main__":
    app.run()
