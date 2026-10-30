import hashlib

import pytest

from src.inference import (
    LocalTransformersAdapter,
    prompt_hash,
    resolve_revision,
)


class _FakeOOM(Exception):
    pass


class _FakeAdapter(LocalTransformersAdapter):
    def __init__(self, batch_size=4, fail_over=2, always_fail=None):
        self.batch_size = batch_size
        self.max_new_tokens = 8
        self.max_retries = 3
        self.model_id = "fake/model"
        self.revision = "rev123"
        self.dtype = "bfloat16"
        self.quantization = "none"
        self.oom_events = []
        self.retry_events = []
        self.batch_size_history = [batch_size]
        self.fail_over = fail_over
        self.always_fail = always_fail
        self.calls = []

    def _encode(self, prompts):
        self.calls.append(len(prompts))
        return {"prompts": list(prompts)}, 1

    def _generate_tokens(self, encoded, max_new_tokens):
        if self.always_fail is not None:
            raise self.always_fail
        if len(encoded["prompts"]) > self.fail_over:
            raise _FakeOOM()
        return [f"out:{prompt}" for prompt in encoded["prompts"]]

    def _decode(self, output, input_len):
        return output

    def _is_oom(self, exc):
        return isinstance(exc, _FakeOOM)

    def _empty_cache(self):
        pass


class _Info:
    sha = "resolved-sha"


class _Hub:
    def model_info(self, model_id):
        return _Info()


class _ExplodingHub:
    def model_info(self, model_id):
        raise AssertionError("pinned revision must not call the hub")


def test_prompt_hash_is_sha256():
    assert prompt_hash("hello") == hashlib.sha256(b"hello").hexdigest()


def test_resolve_revision_prefers_the_pin():
    assert resolve_revision("model", "abc123", _ExplodingHub()) == "abc123"
    assert resolve_revision("model", None, _Hub()) == "resolved-sha"
    assert resolve_revision("model", "", _Hub()) == "resolved-sha"


def test_only_unquantized_models_are_supported():
    with pytest.raises(NotImplementedError):
        LocalTransformersAdapter(quantization="8bit")


def test_batch_generation_preserves_order_and_metadata():
    adapter = _FakeAdapter(batch_size=4, fail_over=10)
    records = adapter.generate_batch(["a", "b", "c"], run_id="run-1", seed=7)
    assert [r.raw for r in records] == ["out:a", "out:b", "out:c"]
    assert [r.prompt_hash for r in records] == [prompt_hash(p) for p in ["a", "b", "c"]]
    assert all(r.run_id == "run-1" for r in records)
    assert all(r.seed == 7 for r in records)
    assert all(r.model_revision == "rev123" for r in records)
    assert all(r.status == "ok" and r.duration_s >= 0 for r in records)


def test_format_prompt_uses_model_chat_template_when_available():
    adapter = _FakeAdapter()

    class _Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "hello"}]
            assert tokenize is False
            assert add_generation_prompt is True
            return "<user>hello<assistant>"

    adapter.tokenizer = _Tokenizer()
    assert adapter._format_prompt("hello") == "<user>hello<assistant>"


def test_empty_batch_returns_empty_list():
    adapter = _FakeAdapter()
    assert adapter.generate_batch([]) == []


def test_generate_wraps_a_single_batch():
    adapter = _FakeAdapter(batch_size=1, fail_over=10)
    record = adapter.generate("only")
    assert record.raw == "out:only"
    assert record.replicate == 1


def test_oom_halves_the_batch_and_retries():
    adapter = _FakeAdapter(batch_size=4, fail_over=2)
    records = adapter.generate_batch(["a", "b", "c", "d"])
    assert [r.raw for r in records] == ["out:a", "out:b", "out:c", "out:d"]
    assert adapter.calls == [4, 2, 2]
    assert adapter.batch_size_history == [4, 2]
    assert adapter.oom_events == [{"event": "oom_halve", "from": 4, "to": 2, "at": 0}]
    assert adapter.batch_size == 4


def test_oom_at_batch_size_one_becomes_error_records():
    adapter = _FakeAdapter(batch_size=1, fail_over=0)
    records = adapter.generate_batch(["a", "b"])
    assert [r.status for r in records] == ["error", "error"]
    assert all(r.raw == "" for r in records)
    assert all("_FakeOOM" in r.usage["error"] for r in records)
    assert len(adapter.retry_events) == 4


def test_non_oom_failures_retry_then_become_error_records():
    adapter = _FakeAdapter(batch_size=2, always_fail=ValueError("boom"))
    records = adapter.generate_batch(["a", "b"])
    assert [r.status for r in records] == ["error", "error"]
    assert all("ValueError: boom" in r.usage["error"] for r in records)
    assert len(adapter.retry_events) == 2
    assert adapter.calls == [2, 2, 2]


def test_environment_record_reports_the_loaded_stack(monkeypatch):
    class _Cuda:
        @staticmethod
        def get_device_name(index):
            return "Fake GPU"

    class _Version:
        cuda = "13.2"

    class _Torch:
        cuda = _Cuda()
        version = _Version()

    adapter = _FakeAdapter()
    adapter.torch = _Torch()
    adapter.serving_library_version = "4.99.0"
    adapter.torch_version = "2.9.0"
    monkeypatch.setattr("src.inference.driver_version", lambda: "595.71.05")

    record = adapter.environment_record()
    assert record["gpu_name"] == "Fake GPU"
    assert record["cuda_version"] == "13.2"
    assert record["driver_version"] == "595.71.05"
    assert record["serving_library_version"] == "4.99.0"
    assert record["torch_version"] == "2.9.0"
    assert record["batch_size"] == 4
    assert record["revision"] == "rev123"
    assert record["oom_events"] == 0


def test_constructor_wires_loading_and_pads_on_the_left():
    class _Tokenizer:
        def __init__(self):
            self.pad_token = None
            self.eos_token = "</s>"
            self.pad_token_id = 0
            self.padding_side = "right"

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, revision):
            assert model_id == "fake/model"
            assert revision == "resolved-sha"
            return _Tokenizer()

    class _Model:
        eval_called = False

        def eval(self):
            self.eval_called = True

    class _AutoModel:
        captured = {}

        @classmethod
        def from_pretrained(cls, model_id, revision, torch_dtype, device_map):
            cls.captured = {
                "model_id": model_id,
                "revision": revision,
                "torch_dtype": torch_dtype,
                "device_map": device_map,
            }
            return _Model()

    class _Transformers:
        __version__ = "4.99.0"
        AutoTokenizer = _AutoTokenizer
        AutoModelForCausalLM = _AutoModel

    class _Cuda:
        @staticmethod
        def is_available():
            return True

    class _Torch:
        __version__ = "2.9.0"
        bfloat16 = "bf16"
        cuda = _Cuda()

    adapter = LocalTransformersAdapter(
        "fake/model",
        revision=None,
        torch_module=_Torch,
        transformers_module=_Transformers,
        hub_api=_Hub(),
    )
    assert adapter.revision == "resolved-sha"
    assert adapter.device == "cuda"
    assert adapter.tokenizer.padding_side == "left"
    assert adapter.tokenizer.pad_token == "</s>"
    assert adapter.model.eval_called is True
    assert _AutoModel.captured == {
        "model_id": "fake/model",
        "revision": "resolved-sha",
        "torch_dtype": "bf16",
        "device_map": "cuda",
    }
    assert adapter.serving_library_version == "4.99.0"
    assert adapter.torch_version == "2.9.0"
