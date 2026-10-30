"""Build the uploadable Molab bundle with project sources and saved results."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "promptrepair_molab_bundle.zip"
INCLUDED_DIRS = ("src", "config", "prompts", "tests", "scripts", "runs", "examples")
INCLUDED_FILES = (
    "README.md",
    "analysis_plan.md",
    "task_contract.md",
    "PROJECT_COMPLETION_PLAN.md",
    "PROJECT_OVERVIEW.md",
    "RESEARCH_IMPROVEMENT_PLAN.md",
    "RESEARCH_BROADER_PROMPT_REPAIR_8H.md",
    "CHAT_HANDOFF.md",
    "project_plan.md",
    "requirements.txt",
    "molab_one_file_runner.py",
    "molab_broad_repair_runner.py",
)
INCLUDED_DATA_FILES = (
    "data/feedback.jsonl",
    "data/selection.jsonl",
    "data/test_inputs.jsonl",
    "data/test_labels.jsonl",
    "data/split_manifest.json",
    "data/exploratory_draft_02/inputs.jsonl",
    "data/exploratory_draft_02/labels_draft.jsonl",
    "data/exploratory_draft_02/REVIEW_GUIDE.md",
    "data/exploratory_draft_02/CHANGELOG.md",
    "data/exploratory_draft_02/human_review.json",
    "data/exploratory_draft_03/inputs.jsonl",
    "data/exploratory_draft_03/labels_proposed.jsonl",
    "data/exploratory_draft_03/labels_reviewed.jsonl",
    "data/exploratory_draft_03/human_review.json",
    "data/exploratory_draft_03/REVIEW_GUIDE.md",
    "data/prompt_repair_multitask/dataset_plan.json",
    "data/prompt_repair_multitask/dataset_sources.json",
    "data/prompt_repair_multitask/ag_news_manifest.json",
    "data/prompt_repair_multitask/squad2_manifest.json",
    "data/prompt_repair_multitask/source_examples_manifest.json",
    "data/prompt_repair_multitask/superni_task_ids_manifest.json",
    "data/prompt_repair_multitask/superni_metadata_sample_manifest.json",
    "data/prompt_repair_multitask/superni_source_examples_manifest.json",
    "data/prompt_repair_multitask/superni_metadata_sample_02_manifest.json",
    "data/prompt_repair_multitask/superni_metadata_remaining_scan_manifest.json",
    "data/prompt_repair_multitask/superni_source_examples_v2_manifest.json",
    "data/prompt_repair_multitask/superni_source_examples_v2_audit_manifest.json",
    "data/prompt_repair_multitask/superni_source_examples_v3_manifest.json",
    "data/prompt_repair_multitask/superni_source_examples_v3_clean_manifest.json",
    "data/prompt_repair_multitask/superni_source_examples_v3_clean_audit_manifest.json",
    "data/prompt_repair_multitask/repair_candidate_cases_manifest.json",
    "data/prompt_repair_multitask/repair_candidate_validation_manifest.json",
    "data/prompt_repair_multitask/repair_candidate_training_screen_manifest.json",
    "data/prompt_repair_multitask/repair_candidate_cases_v3_manifest.json",
    "data/prompt_repair_multitask/README.md",
)
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache"}
EXCLUDED_FILES = {"prepare_exploratory_v2.py"}


def included_paths() -> list[Path]:
    paths: set[Path] = set()
    for relative_dir in INCLUDED_DIRS:
        directory = ROOT / relative_dir
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if (
                not path.is_file()
                or EXCLUDED_PARTS.intersection(path.parts)
                or path.name in EXCLUDED_FILES
            ):
                continue
            paths.add(path)
    paths.update(ROOT / relative for relative in INCLUDED_FILES + INCLUDED_DATA_FILES)
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing bundle inputs: {sorted(missing)}")
    return sorted(paths, key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> int:
    paths = included_paths()
    manifest = {
        "format": 1,
        "purpose": "Molab restart bundle: source, frozen inputs, and saved run results",
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in paths
        ],
    }
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr(
            "bundle_manifest.json",
            json.dumps(manifest, indent=2) + "\n",
        )
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"Wrote {OUTPUT}")
    print(f"Files: {len(paths)}")
    print(f"Size: {OUTPUT.stat().st_size} bytes")
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
