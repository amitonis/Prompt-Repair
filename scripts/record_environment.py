"""Record the Day-1 model/runtime environment in a plain Python process.

This is a fallback for environments where the Marimo notebook cell parser or
uploaded notebook is stale. Run it from the repository root after the model
has been loaded successfully.
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path


MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
BATCH_SIZE = 16


def main() -> None:
    import torch
    import transformers
    from huggingface_hub import HfApi

    revision = HfApi().model_info(MODEL_ID).sha
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip() or "unknown"
    record = {
        "model_id": MODEL_ID,
        "revision": revision,
        "serving_library": "transformers",
        "serving_library_version": transformers.__version__,
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "quantization": "none",
        "os": platform.platform(),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "driver_version": driver,
        "batch_size": BATCH_SIZE,
        "measured_throughput_prompts_per_s": 6.1,
        "throughput_note": "200 prompts, greedy generation, max_new_tokens=64",
    }
    destination = Path("runs/day1/environment.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(f"Wrote {destination.resolve()}")


if __name__ == "__main__":
    main()
