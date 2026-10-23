from src.experiment_runner import cache_key


def test_cache_key_is_deterministic():
    a = cache_key("model-x", "rev1", "vllm 0.6", None, "prompt text", "notice text", 42)
    b = cache_key("model-x", "rev1", "vllm 0.6", None, "prompt text", "notice text", 42)
    assert a == b


def test_cache_key_changes_with_each_component():
    base = cache_key("model-x", "rev1", "vllm 0.6", None, "prompt", "notice", 42)
    assert cache_key("model-y", "rev1", "vllm 0.6", None, "prompt", "notice", 42) != base
    assert cache_key("model-x", "rev2", "vllm 0.6", None, "prompt", "notice", 42) != base
    assert cache_key("model-x", "rev1", "vllm 0.7", None, "prompt", "notice", 42) != base
    assert cache_key("model-x", "rev1", "vllm 0.6", "8bit", "prompt", "notice", 42) != base
    assert cache_key("model-x", "rev1", "vllm 0.6", None, "prompt2", "notice", 42) != base
    assert cache_key("model-x", "rev1", "vllm 0.6", None, "prompt", "notice2", 42) != base
    assert cache_key("model-x", "rev1", "vllm 0.6", None, "prompt", "notice", 43) != base
