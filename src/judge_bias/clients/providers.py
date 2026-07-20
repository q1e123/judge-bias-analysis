from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from judge_bias.clients.base import Completion, Message, messages_to_openai
from judge_bias.config import ModelSpec


def _api_params(params: dict[str, Any]) -> dict[str, Any]:
    """Drop harness-only control keys (``_ctx`` etc.) before calling the API."""
    return {k: v for k, v in params.items() if not k.startswith("_")}


def _key(spec: ModelSpec, default_env: str) -> str | None:
    return os.environ.get(spec.api_key_env or default_env)


def _parse_retry_wait(msg: str) -> float | None:
    """Total suggested wait (seconds) from a rate-limit message.

    Handles compound forms like 'try again in 15m14.112s' and 'in 410ms'.
    """
    import re

    m = re.search(r"try again in\s+([0-9hms.\s]+)", msg, re.IGNORECASE)
    if not m:
        return None
    unit_s = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    total = 0.0
    found = False
    for val, unit in re.findall(r"([\d.]+)\s*(ms|h|m|s)", m.group(1), re.IGNORECASE):
        total += float(val) * unit_s[unit.lower()]
        found = True
    return total if found else None


_MAX_WAIT_S = 45.0


_SERVER_ERR = ("internalservererror", "internal server error", "500", "502",
               "503", "504", "bad gateway", "service unavailable", "overloaded",
               "timeout", "timed out", "connection")


def _is_server_error(msg: str) -> bool:
    m = msg.lower()
    return any(tok in m for tok in _SERVER_ERR)


def _create_with_retry(create_fn, *, max_retries: int = 6):
    for attempt in range(max_retries):
        try:
            return create_fn()
        except Exception as e:  # noqa: BLE001 - SDK-specific error types vary
            msg = str(e)
            rate_limited = ("rate_limit" in msg or "429" in msg
                            or "try again in" in msg.lower())
            server_err = _is_server_error(msg)
            wait = _parse_retry_wait(msg)
            last = attempt == max_retries - 1
            too_long = rate_limited and wait is not None and wait > _MAX_WAIT_S
            if last or too_long or not (rate_limited or server_err):
                raise
            if rate_limited and wait is not None:
                time.sleep(wait + 0.3)
            else:  # server error / unspecified wait: back off, leave load time
                time.sleep(min(5 * (attempt + 1), 40))


class OpenAICompatClient:

    def __init__(self, spec: ModelSpec):
        from openai import OpenAI  # lazy - used only as an HTTP client

        self.spec = spec
        key = _key(spec, "OPENWEBUI_API_KEY")
        if not key and spec.base_url:
            key = "not-needed"
        self._client = OpenAI(api_key=key, base_url=spec.base_url or None)

    def complete(self, messages: list[Message], **params: Any) -> Completion:
        ap = _api_params(params)
        t0 = time.perf_counter()

        def _run():
            stream = self._client.chat.completions.create(
                model=self.spec.model_id,
                messages=messages_to_openai(messages),
                stream=True,
                **ap,
            )
            parts: list[str] = []
            usage = None
            for chunk in stream:
                if getattr(chunk, "usage", None):  # final chunk carries usage
                    usage = chunk.usage
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        parts.append(delta.content)
            return "".join(parts), usage

        text, u = _create_with_retry(_run)
        dt = time.perf_counter() - t0
        return Completion(
            text=text,
            tokens_in=getattr(u, "prompt_tokens", 0) if u else 0,
            tokens_out=getattr(u, "completion_tokens", 0) if u else 0,
            latency_s=dt,
            raw=None,
        )


class GroqClient:
    """Groq exposes an OpenAI-compatible chat API."""

    def __init__(self, spec: ModelSpec):
        from groq import Groq  # lazy

        self.spec = spec
        kwargs = {"api_key": _key(spec, "GROQ_API_KEY")}
        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        self._client = Groq(**kwargs)

    def complete(self, messages: list[Message], **params: Any) -> Completion:
        ap = _api_params(params)
        t0 = time.perf_counter()
        resp = _create_with_retry(lambda: self._client.chat.completions.create(
            model=self.spec.model_id,
            messages=messages_to_openai(messages),
            **ap,
        ))
        dt = time.perf_counter() - t0
        u = resp.usage
        return Completion(
            text=resp.choices[0].message.content or "",
            tokens_in=getattr(u, "prompt_tokens", 0),
            tokens_out=getattr(u, "completion_tokens", 0),
            latency_s=dt,
            raw=resp,
        )


class OllamaClient:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        host = spec.base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.base = host.rstrip("/")

    def complete(self, messages: list[Message], **params: Any) -> Completion:
        ap = _api_params(params)
        payload = {
            "model": self.spec.model_id,
            "messages": messages_to_openai(messages),
            "stream": False,
        }
        if ap:
            payload["options"] = ap  # temperature, num_predict, seed, ...
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/api/chat", data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:  # daemon down / model missing
            raise RuntimeError(
                f"Ollama request to {self.base} failed for model "
                f"{self.spec.model_id!r}: {e}. Is `ollama serve` running and the "
                f"model pulled (`ollama pull {self.spec.model_id}`)?"
            ) from e
        dt = time.perf_counter() - t0
        return Completion(
            text=body.get("message", {}).get("content", ""),
            tokens_in=body.get("prompt_eval_count", 0),
            tokens_out=body.get("eval_count", 0),
            latency_s=dt,
            raw=body,
        )
