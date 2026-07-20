"""Reproducibility manifest for the released artifact.

Records exactly what produced a result set: model registry (ids/providers),
dataset item hashes + leakage tags, library versions, seed, and timestamps.
Shipping this alongside the judgment JSONL is what lets a reviewer reproduce -
and lets the paper state model snapshots and dataset versions precisely.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from judge_bias import __version__
from judge_bias.config import RunConfig
from judge_bias.dataset.schema import TaskItem

_TRACKED_LIBS = ["numpy", "pandas", "scipy", "statsmodels", "scikit-learn",
                 "matplotlib", "krippendorff", "pydantic"]


def _lib_versions() -> dict[str, str]:
    out = {}
    for lib in _TRACKED_LIBS:
        try:
            out[lib] = version(lib)
        except PackageNotFoundError:
            out[lib] = "not-installed"
    return out


def items_digest(items: list[TaskItem]) -> dict:
    """Stable hash over item ids + prompts, plus a leakage-tag census."""
    h = hashlib.sha256()
    leakage: dict[str, int] = {}
    for it in sorted(items, key=lambda x: (x.dataset, x.item_id)):
        h.update(f"{it.dataset}|{it.item_id}|{it.prompt}".encode())
        leakage[it.leakage.value] = leakage.get(it.leakage.value, 0) + 1
    return {"n_items": len(items), "sha256": h.hexdigest(), "leakage": leakage}


def build_manifest(cfg: RunConfig, items: list[TaskItem],
                   extra: dict | None = None) -> dict:
    return {
        "judge_bias_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": cfg.seed,
        "libraries": _lib_versions(),
        "models": [
            {"name": m.name, "provider": m.provider, "model_id": m.model_id,
             "family": m.family, "size_b": m.size_b, "role": m.role,
             "params": m.params}
            for m in cfg.models
        ],
        "datasets": items_digest(items),
        **(extra or {}),
    }


def write_manifest(cfg: RunConfig, items: list[TaskItem],
                   extra: dict | None = None) -> Path:
    cfg.ensure_dirs()
    path = cfg.results_dir / "manifest.json"
    path.write_text(json.dumps(build_manifest(cfg, items, extra), indent=2),
                    encoding="utf-8")
    return path
