"""Provider-agnostic chat client interface."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Completion:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    raw: Any = None


class ChatClient(Protocol):
    """Minimal contract every provider wrapper implements."""

    def complete(self, messages: list[Message], **params: Any) -> Completion: ...


class _Timer:
    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t


def messages_to_openai(messages: list[Message]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]
