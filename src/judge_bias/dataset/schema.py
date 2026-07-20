"""Common schema shared across every dataset and every pipeline stage.

One canonical record type per concept, all JSON-serialisable via pydantic.
The JSONL judgment record in :class:`Judgment` is the reproducibility artifact.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LeakageTag(str, Enum):
    """Whether an item was plausibly in a model's training corpus (leakage note)."""

    LIKELY_SEEN = "likely_seen"
    UNCERTAIN = "uncertain"
    LIKELY_UNSEEN = "likely_unseen"


class Rubric(BaseModel):
    """Scoring rubric handed to a pointwise judge. Mirrors FARR-Flow 0-3 scales."""

    name: str = "outcome"
    scale_min: int = 0
    scale_max: int = 3
    description: str = (
        "Score the response 0-3 on how well it achieves the requested objective."
    )
    anchors: dict[int, str] = Field(default_factory=dict)


class ReferenceLabel(BaseModel):
    """Human ground-truth anchor"""

    chosen: str | None = None
    rejected: str | None = None
    gold_score: float | None = None
    source: str | None = None


class TaskItem(BaseModel):
    """A single prompt to be answered by solvers and scored by judges."""

    item_id: str
    dataset: str
    category: str | None = None
    prompt: str
    rubric: Rubric | None = None
    reference: ReferenceLabel | None = None
    leakage: LeakageTag = LeakageTag.UNCERTAIN
    refusal_relevant: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


class Response(BaseModel):
    """A solver's answer to one :class:`TaskItem`."""

    item_id: str
    dataset: str
    solver: str
    text: str
    refused: bool = False
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.dataset, self.item_id, self.solver)


class JudgeMode(str, Enum):
    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"


class Condition(str, Enum):
    """Experimental condition a judgment was produced under."""

    NAIVE = "naive"          # plain single-judge baseline
    BLINDED = "blinded"      # identity / style tells stripped
    SWAPPED = "swapped"      # pairwise run in reversed order


class Judgment(BaseModel):
    """The canonical JSONL judgment record.
    """

    dataset: str
    item_id: str
    judge: str
    mode: JudgeMode
    condition: Condition = Condition.NAIVE
    prompt_variant: str = "default"      # judge-prompt template used (robustness sweep)

    # Pointwise fields
    solver: str | None = None
    parsed_score: float | None = None

    # Pairwise fields
    solver_a: str | None = None
    solver_b: str | None = None
    order: str | None = None          # "AB" or "BA" as presented to the judge
    winner: str | None = None         # solver name of the chosen answer, or "tie"

    # Diagnostics
    raw_output: str = ""
    refused: bool = False                # judge refused / unparseable
    solver_refused: bool = False         # the judged answer was itself a refusal
    answer_len: int | None = None     # length of the judged answer (verbosity)
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    def key(self) -> tuple:
        """Unique tuple used for caching / resume (skip if already present)."""
        if self.mode == JudgeMode.POINTWISE:
            return (self.dataset, self.item_id, self.judge, "pw", self.solver,
                    self.condition.value, self.prompt_variant)
        return (self.dataset, self.item_id, self.judge, "pair", self.solver_a,
                self.solver_b, self.order, self.condition.value)
