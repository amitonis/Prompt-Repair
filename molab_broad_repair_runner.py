# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo"]
# ///
"""Fresh Molab bootstrap for the broader cross-task prompt-repair work.

Upload promptrepair_molab_bundle.zip in the first UI cell. The runner extracts
it into a separate promptrepair_broad_runtime directory, verifies the bundle
manifest, and reports runtime readiness. It does not train or load a model.
"""

from __future__ import annotations

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # PromptRepair — fresh Molab bootstrap

        This notebook starts a separate runtime for the broader, multi-task
        prompt-repair work. It will **not** reuse the older notice-task
        notebook's Python modules or output folder.

        1. Upload the current `promptrepair_molab_bundle.zip` below.
        2. Run the extraction and hash-verification cell.
        3. Run the environment preflight cell.

        These setup cells do not load model weights or start training. The
        cross-task repair dataset and its task-held-out splits must be prepared
        before an 8–10-hour adapter-training run.
        """
    )
    return


@app.cell
def _(mo):
    broad_bundle_upload = mo.ui.file(
        filetypes=[".zip"],
        multiple=False,
        kind="area",
        label="Upload promptrepair_molab_bundle.zip",
    )
    broad_bundle_upload
    return (broad_bundle_upload,)


@app.cell
def _(broad_bundle_upload, mo):
    import hashlib as broad_hashlib
    import io as broad_io
    import json as broad_json
    import sys as broad_sys
    import zipfile as broad_zipfile
    from pathlib import Path as BroadPath

    broad_runtime_root = BroadPath("promptrepair_broad_runtime").resolve()
    broad_bundle_status = "Waiting for the project ZIP upload."
    broad_bundle_record = None

    if broad_bundle_upload.value:
        broad_upload_name = broad_bundle_upload.name() or "uploaded_bundle.zip"
        broad_upload_bytes = broad_bundle_upload.contents()
        if broad_upload_bytes is None:
            raise ValueError("The uploaded ZIP has no readable contents.")
        if not broad_upload_name.lower().endswith(".zip"):
            raise ValueError("Upload promptrepair_molab_bundle.zip.")

        broad_runtime_root.mkdir(parents=True, exist_ok=True)
        with broad_zipfile.ZipFile(broad_io.BytesIO(broad_upload_bytes)) as broad_archive:
            for broad_member in broad_archive.infolist():
                broad_target = (broad_runtime_root / broad_member.filename).resolve()
                if broad_runtime_root not in broad_target.parents:
                    raise ValueError(f"Unsafe ZIP member path: {broad_member.filename}")
            broad_archive.extractall(broad_runtime_root)

        broad_manifest_path = broad_runtime_root / "bundle_manifest.json"
        if not broad_manifest_path.is_file():
            raise FileNotFoundError("The archive has no bundle_manifest.json.")
        broad_manifest = broad_json.loads(
            broad_manifest_path.read_text(encoding="utf-8")
        )
        broad_verified_files = 0
        broad_hash_failures = []
        for broad_entry in broad_manifest.get("files", []):
            broad_file = (broad_runtime_root / broad_entry["path"]).resolve()
            if broad_runtime_root not in broad_file.parents or not broad_file.is_file():
                broad_hash_failures.append(broad_entry["path"])
                continue
            broad_actual_hash = broad_hashlib.sha256(broad_file.read_bytes()).hexdigest()
            if broad_actual_hash != broad_entry["sha256"]:
                broad_hash_failures.append(broad_entry["path"])
            else:
                broad_verified_files += 1
        if broad_hash_failures:
            raise RuntimeError(
                "Bundle hash verification failed for: "
                + ", ".join(broad_hash_failures[:10])
            )

        broad_archive_hash = broad_hashlib.sha256(broad_upload_bytes).hexdigest()
        broad_bundle_record = {
            "uploaded_name": broad_upload_name,
            "archive_sha256": broad_archive_hash,
            "manifest_files_verified": broad_verified_files,
            "runtime_root": str(broad_runtime_root),
        }
        (broad_runtime_root / "runs").mkdir(parents=True, exist_ok=True)
        (broad_runtime_root / "runs" / "broad_runner_bundle_metadata.json").write_text(
            broad_json.dumps(broad_bundle_record, indent=2) + "\n",
            encoding="utf-8",
        )
        if str(broad_runtime_root) not in broad_sys.path:
            broad_sys.path.insert(0, str(broad_runtime_root))
        broad_bundle_status = (
            f"Extracted and verified {broad_verified_files} files into "
            f"`{broad_runtime_root}`. Archive SHA-256: `{broad_archive_hash}`."
        )

    mo.md(broad_bundle_status)
    return broad_bundle_record, broad_bundle_status, broad_runtime_root


@app.cell
def _(broad_runtime_root, mo):
    import importlib.util as broad_import_util
    import json as broad_preflight_json

    broad_modules = {
        name: broad_import_util.find_spec(name) is not None
        for name in ("torch", "transformers", "accelerate", "peft", "trl", "bitsandbytes", "datasets")
    }
    broad_preflight = {
        "runtime_root_exists": broad_runtime_root.is_dir(),
        "repair_source_exists": (broad_runtime_root / "src" / "reviewer_tool.py").is_file(),
        "multitask_repair_data_exists": (
            broad_runtime_root / "data" / "prompt_repair_multitask"
        ).exists(),
        "python_packages_available": broad_modules,
        "cuda_available": False,
        "gpu_name": None,
        "free_gpu_memory_gib": None,
        "total_gpu_memory_gib": None,
    }

    if broad_preflight["runtime_root_exists"]:
        try:
            import torch as broad_torch

            broad_preflight["torch_version"] = broad_torch.__version__
            broad_preflight["cuda_available"] = broad_torch.cuda.is_available()
            if broad_preflight["cuda_available"]:
                broad_free_bytes, broad_total_bytes = broad_torch.cuda.mem_get_info()
                broad_preflight["gpu_name"] = broad_torch.cuda.get_device_name(0)
                broad_preflight["free_gpu_memory_gib"] = round(
                    broad_free_bytes / 1024**3, 2
                )
                broad_preflight["total_gpu_memory_gib"] = round(
                    broad_total_bytes / 1024**3, 2
                )
        except Exception as broad_torch_error:
            broad_preflight["torch_check_error"] = str(broad_torch_error)

        try:
            import transformers as broad_transformers

            broad_preflight["transformers_version"] = broad_transformers.__version__
        except Exception as broad_transformers_error:
            broad_preflight["transformers_import_error"] = str(
                broad_transformers_error
            )

    broad_preflight["ready_for_adapter_training"] = bool(
        broad_preflight["runtime_root_exists"]
        and broad_preflight["cuda_available"]
        and broad_modules["accelerate"]
        and broad_modules["peft"]
        and broad_modules["trl"]
        and broad_modules["bitsandbytes"]
        and broad_modules["datasets"]
        and broad_preflight["multitask_repair_data_exists"]
    )

    mo.md(
        "### Runtime preflight\n\n```json\n"
        + broad_preflight_json.dumps(broad_preflight, indent=2)
        + "\n```\n\n"
        + (
            "Runtime and cross-task dataset are present."
            if broad_preflight["ready_for_adapter_training"]
            else "Setup only: training should wait until missing packages and the cross-task repair dataset are ready."
        )
    )
    return (broad_preflight,)


if __name__ == "__main__":
    app.run()
