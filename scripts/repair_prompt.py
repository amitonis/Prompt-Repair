"""Command-line entry point for the Tier A reviewer prompt-repair tool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reviewer_tool import run_repair


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Repair a four-section or plain-text prompt for the fixed college-notice "
            "task. Uses development feedback and selection data only; never loads test data."
        )
    )
    parser.add_argument("prompt_file", help="JSON sectioned prompt or plain-text instruction file")
    parser.add_argument("--mode", choices=("targeted", "rewrite", "both"), default="both")
    parser.add_argument("--output-dir", help="New output directory; existing directories are never overwritten")
    parser.add_argument("--run-id", help="Stable artifact name; must be unique")
    parser.add_argument("--device", default="cuda", help="Model device (default: cuda)")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--max-attempts", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)

    output = run_repair(
        args.prompt_file,
        mode=args.mode,
        output_dir=args.output_dir,
        run_id=args.run_id,
        device=args.device,
        seed=args.seed,
        max_attempts=args.max_attempts,
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    print(f"Saved report: {output / 'report.md'}")
    for method, result in report["repairs"].items():
        metrics = result["selection_metrics_after"]
        print(
            f"{method}: {metrics['record_correct']}/{metrics['attempted']} selection records; "
            f"accepted={result['accepted']}; changed={result['changed_sections'] or 'none'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
