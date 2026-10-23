"""Model adapter: locally served open-weight models, optional hosted control.

Contract (Section 13.3): accepts a rendered prompt and returns a record with
response text, status, model metadata, usage, and duration. The rest of the
application must not know which backend produced a response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationRecord:
    run_id: str
    prompt_hash: str
    model_id: str
    model_revision: str | None
    raw: str
    status: str
    duration_s: float
    usage: dict[str, Any] = field(default_factory=dict)
    serving_library: str | None = None
    quantization: str | None = None
    seed: int | None = None
    replicate: int = 1
    cached: bool = False


class ModelAdapter:
    """Base class. Implement with vLLM/Transformers batching on molab (Day 1-5)."""

    def generate(self, rendered_prompt: str, **settings: Any) -> GenerationRecord:
        raise NotImplementedError("Local serving adapter is implemented on molab (Day 5)")

    def generate_batch(self, rendered_prompts: list[str], **settings: Any) -> list[GenerationRecord]:
        raise NotImplementedError("Local serving adapter is implemented on molab (Day 5)")


class HostedApiAdapter(ModelAdapter):
    """Optional control arm only (Section 14.9). Track its calls separately."""

    def generate(self, rendered_prompt: str, **settings: Any) -> GenerationRecord:
        raise NotImplementedError("Optional hosted-API control is not part of the core run")
