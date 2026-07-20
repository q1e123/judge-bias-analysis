from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    name: str
    provider: str
    model_id: str
    family: str
    size_b: float | None = None
    active_size_b: float | None = None
    role: str = "solver+judge"
    params: dict[str, Any] = Field(default_factory=lambda: {"temperature": 0.0})
    size_rank: float | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    notes: str = ""

    @property
    def is_solver(self) -> bool:
        return "solver" in self.role

    @property
    def is_judge(self) -> bool:
        return "judge" in self.role

    def size_for_regression(self, measure: str = "total") -> float | None:
        """Numeric total- or active-parameter proxy for size regressions."""
        if measure == "active":
            return self.active_size_b or self.size_b or self.size_rank
        if measure != "total":
            raise ValueError(f"unknown size measure {measure!r}")
        if self.size_b is not None:
            return self.size_b
        return self.size_rank


class RunConfig(BaseModel):
    """Top-level configuration for a study run."""

    models: list[ModelSpec]
    workdir: Path = Path("workspace")
    seed: int = 7
    run_blinded: bool = True
    run_swapped: bool = True
    pairwise_fraction: float = 1.0
    max_retries: int = 3

    model_config = {"arbitrary_types_allowed": True}

    @property
    def solvers(self) -> list[ModelSpec]:
        return [m for m in self.models if m.is_solver]

    @property
    def judges(self) -> list[ModelSpec]:
        return [m for m in self.models if m.is_judge]

    @property
    def cache_dir(self) -> Path:
        return self.workdir / "cache"

    @property
    def results_dir(self) -> Path:
        return self.workdir / "results"

    @property
    def figures_dir(self) -> Path:
        return self.workdir / "figures"

    def ensure_dirs(self) -> None:
        for d in (self.cache_dir, self.results_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)


def groq_registry() -> list[ModelSpec]:
    return [
        ModelSpec(name="llama-3-70b", provider="groq",
                  model_id="llama-3.3-70b-versatile", family="meta",
                  size_b=70.0, notes="FARR-Flow judge"),
        ModelSpec(name="qwen3-32b", provider="groq",
                  model_id="qwen/qwen3-32b", family="alibaba",
                  size_b=32.0, notes="alibaba mid-size"),
        ModelSpec(name="llama4-scout-17b", provider="groq",
                  model_id="meta-llama/llama-4-scout-17b-16e-instruct", family="meta",
                  size_b=17.0, notes="MoE mid-size"),
        ModelSpec(name="llama-3-8b", provider="groq",
                  model_id="llama-3.1-8b-instant", family="meta",
                  size_b=8.0, notes="small-model anchor"),
    ]


def opensource_registry() -> list[ModelSpec]:
    owui = os.environ.get("OPENWEBUI_BASE_URL", "http://localhost:3000/api")
    return [
        ModelSpec(name="llama-3.3-70b", provider="groq",
                  model_id="llama-3.3-70b-versatile", family="meta",
                  size_b=70.0, notes="Groq - FARR-Flow judge, large open"),
        ModelSpec(name="qwen3-32b", provider="groq", model_id="qwen/qwen3-32b",
                  family="alibaba", size_b=32.0, notes="Groq - Alibaba mid-large"),
        ModelSpec(name="llama4-scout-17b", provider="groq",
                  model_id="meta-llama/llama-4-scout-17b-16e-instruct",
                  family="meta", size_b=17.0, notes="Groq - MoE mid-size"),
        ModelSpec(name="mistral-small", provider="openwebui",
                  model_id="mistral-small:latest", family="mistral", size_b=22.0,
                  base_url=owui, api_key_env="OPENWEBUI_API_KEY",
                  notes="OpenWebUI - Mistral family (verify model tag)"),
        ModelSpec(name="deepseek-r1-14b", provider="openwebui",
                  model_id="deepseek-r1:14b", family="deepseek", size_b=14.0,
                  base_url=owui, api_key_env="OPENWEBUI_API_KEY",
                  notes="OpenWebUI - reasoning-distilled (verify model tag)"),
        ModelSpec(name="gemma2-9b", provider="ollama", model_id="gemma2:9b",
                  family="google", size_b=9.0, notes="Ollama local - Google family"),
        ModelSpec(name="llama3.1-8b", provider="ollama", model_id="llama3.1:8b",
                  family="meta", size_b=8.0, notes="Ollama local - size anchor"),
        ModelSpec(name="qwen2.5-7b", provider="ollama", model_id="qwen2.5:7b",
                  family="alibaba", size_b=7.0, notes="Ollama local - size anchor"),
        ModelSpec(name="phi3.5", provider="ollama", model_id="phi3.5:latest",
                  family="microsoft", size_b=3.8, notes="Ollama local - Microsoft family"),
    ]


def default_registry(pool: str = "oss") -> list[ModelSpec]:
    if pool == "groq":
        return groq_registry()
    if pool in ("oss", "opensource", "open-source"):
        return opensource_registry()
    raise ValueError(f"unknown pool {pool!r}; use 'groq' or 'oss'")


def load_registry(path: str | Path) -> list[ModelSpec]:
    """Load a registry from a JSON list of ModelSpec dicts."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ModelSpec(**d) for d in data]
