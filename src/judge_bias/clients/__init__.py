from __future__ import annotations

from judge_bias.clients.base import Completion, Message
from judge_bias.config import ModelSpec

__all__ = ["Completion", "Message", "get_client"]


def get_client(spec: ModelSpec):
    provider = spec.provider
    if provider == "groq":
        from judge_bias.clients.providers import GroqClient

        return GroqClient(spec)
    if provider == "ollama":
        from judge_bias.clients.providers import OllamaClient

        return OllamaClient(spec)
    if provider in ("openwebui", "openai_compat", "local", "vllm"):
        from judge_bias.clients.providers import OpenAICompatClient

        return OpenAICompatClient(spec)
    raise ValueError(
        f"unknown provider {provider!r}; use groq | ollama | openwebui")
