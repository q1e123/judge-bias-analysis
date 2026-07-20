"""Append-only JSONL stores with tuple-keyed resume.

Every solver answer and every judgment is written as one JSON line. Runs are
resumable: load the existing keys, skip anything already present, append the
rest. This makes the expensive NxNxitems matrix safe to interrupt and restart.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from judge_bias.dataset.schema import Judgment, Response


class JsonlStore:
    """A thin append-only JSONL store for a single pydantic model type."""

    def __init__(self, path: str | Path, model_cls):
        self.path = Path(path)
        self.model_cls = model_cls
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return sum(1 for _ in self.iter())

    def iter(self) -> Iterator:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield self.model_cls.model_validate_json(line)

    def load(self) -> list:
        return list(self.iter())

    def append(self, record) -> None:
        # Ensure the dir exists on every write (a wiped/rotated workspace or a
        # race can remove it between construction and the first append).
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")

    def extend(self, records: Iterable) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for r in records:
                fh.write(r.model_dump_json() + "\n")


class ResponseStore(JsonlStore):
    def __init__(self, path: str | Path):
        super().__init__(path, Response)

    def existing_keys(self) -> set[tuple]:
        return {r.key for r in self.iter()}


class JudgmentStore(JsonlStore):
    def __init__(self, path: str | Path):
        super().__init__(path, Judgment)

    def existing_keys(self) -> set[tuple]:
        return {r.key() for r in self.iter()}
