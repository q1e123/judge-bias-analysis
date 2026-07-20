"""Provider routing / registry tests that don't touch the network."""

from __future__ import annotations

import pytest

from judge_bias.clients import get_client
from judge_bias.config import ModelSpec, default_registry


def test_oss_pool_spans_three_open_backends():
    reg = default_registry("oss")
    providers = {m.provider for m in reg}
    assert {"groq", "openwebui", "ollama"} <= providers
    families = {m.family for m in reg}
    # >=4 open families + a small model - separates family bias from size bias (RQ2)
    assert {"meta", "alibaba", "mistral", "google"} <= families
    assert any(m.size_b and m.size_b <= 8 for m in reg)
    # no proprietary providers remain
    assert not ({"openai", "anthropic", "google"} & providers)


def test_openwebui_routes_to_openai_compat_client():
    from judge_bias.clients.providers import OpenAICompatClient

    spec = ModelSpec(name="ms", provider="openwebui", model_id="mistral-small:latest",
                     family="mistral", size_b=22.0, base_url="http://localhost:3000/api",
                     api_key_env="OPENWEBUI_API_KEY")
    assert isinstance(get_client(spec), OpenAICompatClient)
def test_unknown_provider_rejected():
    spec = ModelSpec(name="x", provider="anthropic", model_id="claude", family="anthropic")
    with pytest.raises(ValueError, match="unknown provider"):
        get_client(spec)


def test_ollama_client_routes_and_resolves_host(monkeypatch):
    from judge_bias.clients.providers import OllamaClient

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    spec = ModelSpec(name="l8b", provider="ollama", model_id="llama3.1:8b",
                     family="meta", size_b=8.0)
    client = get_client(spec)
    assert isinstance(client, OllamaClient)
    assert client.base == "http://localhost:11434"

    spec2 = ModelSpec(name="l8b2", provider="ollama", model_id="llama3.1:8b",
                      family="meta", base_url="http://gpu-box:11434/")
    assert get_client(spec2).base == "http://gpu-box:11434"


def test_ollama_error_is_actionable():
    # Unreachable host -> RuntimeError mentioning ollama pull / serve.
    spec = ModelSpec(name="x", provider="ollama", model_id="nope:1b",
                     family="meta", base_url="http://127.0.0.1:1")
    from judge_bias.clients import Message

    with pytest.raises(RuntimeError, match="ollama"):
        get_client(spec).complete([Message("user", "hi")])


def test_api_params_strip_control_keys():
    from judge_bias.clients.providers import _api_params

    assert _api_params({"temperature": 0.0, "_ctx": {"kind": "judge"}}) == {"temperature": 0.0}
