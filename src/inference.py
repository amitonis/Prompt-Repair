"""Model adapters: locally served open-weight models, optional hosted control.

Contract (Section 13.3): accepts a rendered prompt and returns a record with
response text, status, model metadata, usage, and duration. The rest of the
application must not know which backend produced a response.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


def prompt_hash(rendered_prompt: str) -> str:
    return hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()


def resolve_revision(model_id: str, revision: str | None, hub_api: Any = None) -> str:
    """Pin the exact model commit: use the configured revision or resolve it once."""
    if revision:
        return revision
    if hub_api is None:
        from huggingface_hub import HfApi

        hub_api = HfApi()
    return hub_api.model_info(model_id).sha


def driver_version() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


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
    """Base class for local and hosted backends."""

    def generate(self, rendered_prompt: str, **settings: Any) -> GenerationRecord:
        raise NotImplementedError

    def generate_batch(self, rendered_prompts: list[str], **settings: Any) -> list[GenerationRecord]:
        raise NotImplementedError


class LocalTransformersAdapter(ModelAdapter):
    """Greedy, fixed-batch generation with the pre-registered settings.

    Left padding, a fixed batch size and a fixed notice order make greedy
    generation deterministic for a fixed batch composition. Out-of-memory
    halves the batch and retries; other failures retry up to ``max_retries``
    attempts and then return records with status ``error``.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        revision: str | None = None,
        dtype: str = "bfloat16",
        quantization: str = "none",
        batch_size: int = 16,
        max_new_tokens: int = 256,
        max_retries: int = 3,
        device: str | None = None,
        torch_module: Any = None,
        transformers_module: Any = None,
        hub_api: Any = None,
    ) -> None:
        if quantization != "none":
            raise NotImplementedError("Only quantization='none' is pre-registered (Section 1)")
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if max_retries < 1:
            raise ValueError("max_retries must be a positive integer")

        self.torch = torch_module if torch_module is not None else _import("torch")
        self.transformers = (
            transformers_module if transformers_module is not None else _import("transformers")
        )
        self.model_id = model_id
        self.revision = resolve_revision(model_id, revision, hub_api)
        self.dtype = dtype
        self.quantization = quantization
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.oom_events: list[dict[str, Any]] = []
        self.retry_events: list[dict[str, Any]] = []
        self.batch_size_history: list[int] = [batch_size]
        self.device = device or ("cuda" if self.torch.cuda.is_available() else "cpu")

        self.tokenizer = self.transformers.AutoTokenizer.from_pretrained(
            model_id, revision=self.revision
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        dtype_value = getattr(self.torch, DTYPES[dtype], None)
        if dtype_value is None:
            raise ValueError(f"torch module has no attribute {DTYPES[dtype]!r}")
        self.model = self.transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=self.revision,
            torch_dtype=dtype_value,
            device_map=self.device,
        )
        self.model.eval()
        self.serving_library_version = getattr(self.transformers, "__version__", None)
        self.torch_version = getattr(self.torch, "__version__", None)

    def _encode(self, prompts: list[str]) -> tuple[Any, int]:
        model_prompts = [self._format_prompt(prompt) for prompt in prompts]
        encoded = self.tokenizer(model_prompts, return_tensors="pt", padding=True)
        encoded = encoded.to(self.device)
        return encoded, int(encoded["input_ids"].shape[1])

    def _format_prompt(self, rendered_prompt: str) -> str:
        """Wrap the task prompt in the model tokenizer's user chat template."""
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if not callable(apply_template):
            return rendered_prompt
        return apply_template(
            [{"role": "user", "content": rendered_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate_tokens(self, encoded: Any, max_new_tokens: int) -> Any:
        return self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
        )

    def _decode(self, output: Any, input_len: int) -> list[str]:
        new_tokens = output[:, input_len:]
        return [self.tokenizer.decode(row, skip_special_tokens=True).strip() for row in new_tokens]

    def _is_oom(self, exc: BaseException) -> bool:
        oom = getattr(getattr(self.torch, "cuda", None), "OutOfMemoryError", None)
        return oom is not None and isinstance(exc, oom)

    def _empty_cache(self) -> None:
        empty = getattr(getattr(self.torch, "cuda", None), "empty_cache", None)
        if empty is not None:
            empty()

    def generate_batch(self, rendered_prompts: list[str], **settings: Any) -> list[GenerationRecord]:
        if not rendered_prompts:
            return []
        max_new_tokens = int(settings.get("max_new_tokens", self.max_new_tokens))
        run_id = str(settings.get("run_id", ""))
        seed = settings.get("seed")
        replicate = int(settings.get("replicate", 1))

        records: list[GenerationRecord] = []
        current_batch = self.batch_size
        start = 0
        while start < len(rendered_prompts):
            batch = rendered_prompts[start : start + current_batch]
            attempts = 0
            while True:
                attempts += 1
                started = time.perf_counter()
                try:
                    encoded, input_len = self._encode(batch)
                    output = self._generate_tokens(encoded, max_new_tokens)
                    texts = self._decode(output, input_len)
                    if len(texts) != len(batch):
                        raise RuntimeError(
                            f"decoded {len(texts)} texts for {len(batch)} prompts"
                        )
                except Exception as exc:
                    elapsed = time.perf_counter() - started
                    if self._is_oom(exc) and current_batch > 1:
                        self._empty_cache()
                        new_batch = max(1, current_batch // 2)
                        self.oom_events.append(
                            {"event": "oom_halve", "from": current_batch, "to": new_batch, "at": start}
                        )
                        self.batch_size_history.append(new_batch)
                        current_batch = new_batch
                        batch = rendered_prompts[start : start + current_batch]
                        attempts = 0
                        continue
                    if attempts < self.max_retries:
                        self.retry_events.append(
                            {"event": "retry", "attempt": attempts, "at": start, "error": repr(exc)}
                        )
                        continue
                    records.extend(
                        self._record(
                            prompt,
                            run_id=run_id,
                            raw="",
                            status="error",
                            duration=elapsed / len(batch),
                            seed=seed,
                            replicate=replicate,
                            usage={"error": f"{type(exc).__name__}: {exc}"},
                        )
                        for prompt in batch
                    )
                    break
                elapsed = time.perf_counter() - started
                records.extend(
                    self._record(
                        prompt,
                        run_id=run_id,
                        raw=text,
                        status="ok",
                        duration=elapsed / len(batch),
                        seed=seed,
                        replicate=replicate,
                    )
                    for prompt, text in zip(batch, texts)
                )
                break
            start += len(batch)
        return records

    def generate(self, rendered_prompt: str, **settings: Any) -> GenerationRecord:
        return self.generate_batch([rendered_prompt], **settings)[0]

    def _record(
        self,
        prompt: str,
        *,
        run_id: str,
        raw: str,
        status: str,
        duration: float,
        seed: Any,
        replicate: int,
        usage: dict[str, Any] | None = None,
    ) -> GenerationRecord:
        return GenerationRecord(
            run_id=run_id,
            prompt_hash=prompt_hash(prompt),
            model_id=self.model_id,
            model_revision=self.revision,
            raw=raw,
            status=status,
            duration_s=duration,
            usage=usage or {},
            serving_library="transformers",
            quantization=self.quantization,
            seed=seed,
            replicate=replicate,
        )

    def environment_record(self) -> dict[str, Any]:
        torch_module = getattr(self, "torch", None)
        cuda = getattr(torch_module, "cuda", None)
        get_name = getattr(cuda, "get_device_name", None)
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "serving_library": "transformers",
            "serving_library_version": getattr(self, "serving_library_version", None),
            "torch_version": getattr(self, "torch_version", None),
            "dtype": self.dtype,
            "quantization": self.quantization,
            "os": platform.platform(),
            "gpu_name": get_name(0) if get_name is not None else None,
            "cuda_version": getattr(getattr(torch_module, "version", None), "cuda", None),
            "driver_version": driver_version(),
            "batch_size": self.batch_size,
            "oom_events": len(self.oom_events),
            "prompt_format": (
                "tokenizer_chat_template_user_turn"
                if callable(getattr(getattr(self, "tokenizer", None), "apply_chat_template", None))
                else "plain_rendered_prompt"
            ),
        }


class HostedApiAdapter(ModelAdapter):
    """Optional control arm only (Section 14.9). Track its calls separately."""

    def generate(self, rendered_prompt: str, **settings: Any) -> GenerationRecord:
        raise NotImplementedError("Optional hosted-API control is not part of the core run")


def _import(module: str) -> Any:
    import importlib

    return importlib.import_module(module)
