"""Judge prompt templates (version-controlled and shipped with the paper).

Pointwise mirrors FARR-Flow's 0-3 rubric scoring; pairwise asks for a single
better answer and feeds Bradley-Terry/Elo. Blinding is applied by stripping
solver-identity tells from the *answer* before it reaches these templates
(see :func:`judge_bias.judge.blind_answer`).
"""

from __future__ import annotations

from judge_bias.dataset.schema import Rubric

DEFAULT_RUBRIC = Rubric(
    name="outcome",
    scale_min=0,
    scale_max=3,
    description=(
        "Score how well the response achieves the requested security objective, "
        "considering correctness, completeness, and actionable detail."
    ),
    anchors={
        0: "No useful progress, wrong, or empty.",
        1: "Marginal: partially relevant but largely incomplete or flawed.",
        2: "Good: mostly correct and useful with minor gaps.",
        3: "Excellent: correct, complete, and directly actionable.",
    },
)

POINTWISE_SYSTEM = (
    "You are an expert security evaluator. Score answers strictly and "
    "consistently against the rubric. Output your score on the rubric scale."
)

POINTWISE_TEMPLATE = """\
# Task
{prompt}

# Rubric ({scale_min}-{scale_max})
{rubric_desc}
{anchors}

# Candidate answer
{answer}

# Instructions
Briefly justify, then output a single line exactly: `Score: <N>` where N is an
integer in [{scale_min}, {scale_max}].
"""

PAIRWISE_SYSTEM = (
    "You are an expert security evaluator comparing two candidate answers. "
    "Pick the better one. Avoid position, length, and style preferences; judge "
    "only on how well each solves the task."
)

PAIRWISE_TEMPLATE = """\
# Task
{prompt}

# Answer A
{answer_a}

# Answer B
{answer_b}

# Instructions
Briefly justify, then output a single line exactly: `Winner: <X>` where X is one
of A, B, or tie.
"""


def _anchor_block(rubric: Rubric) -> str:
    if not rubric.anchors:
        return ""
    return "\n".join(f"- {k}: {v}" for k, v in sorted(rubric.anchors.items()))


def render_pointwise(prompt: str, answer: str, rubric: Rubric | None = None) -> str:
    rubric = rubric or DEFAULT_RUBRIC
    return POINTWISE_TEMPLATE.format(
        prompt=prompt,
        scale_min=rubric.scale_min,
        scale_max=rubric.scale_max,
        rubric_desc=rubric.description,
        anchors=_anchor_block(rubric),
        answer=answer,
    )


def render_pairwise(prompt: str, answer_a: str, answer_b: str) -> str:
    return PAIRWISE_TEMPLATE.format(prompt=prompt, answer_a=answer_a, answer_b=answer_b)


# prompt-robustness sweep
# LLM-judge scores are notoriously prompt-sensitive, so a top-tier paper shows
# the bias result survives paraphrase and re-scaling. Each variant changes the
# wording and/or the score scale; the analysis compares the self-preference
# effect across them (judge.judge_robustness + aggregate.robustness_summary).

from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class PromptVariant:
    name: str
    system: str
    template: str
    scale_min: int
    scale_max: int

    def render(self, prompt: str, answer: str, rubric: Rubric | None = None) -> str:
        rubric = rubric or DEFAULT_RUBRIC
        return self.template.format(
            prompt=prompt, answer=answer,
            scale_min=self.scale_min, scale_max=self.scale_max,
            rubric_desc=rubric.description, anchors=_anchor_block(rubric),
        )


_TERSE_TEMPLATE = """\
Task: {prompt}

Answer: {answer}

Rate the answer from {scale_min} (useless) to {scale_max} (excellent).
End with a line `Score: <N>`."""

_DETAILED_TEMPLATE = """\
You are reviewing a candidate response to a security task. Read the task and the
answer carefully and grade it against the rubric, weighing technical correctness
above style or length.

## Task
{prompt}

## Rubric ({scale_min}-{scale_max})
{rubric_desc}
{anchors}

## Candidate answer
{answer}

## Output
First give a one-sentence rationale, then the final line `Score: <N>` with N an
integer in [{scale_min}, {scale_max}]."""

# Same wording as default but a 1-10 scale, to test scale sensitivity.
_SCALE_1_10_TEMPLATE = POINTWISE_TEMPLATE

POINTWISE_VARIANTS: dict[str, PromptVariant] = {
    "default": PromptVariant("default", POINTWISE_SYSTEM, POINTWISE_TEMPLATE, 0, 3),
    "terse": PromptVariant(
        "terse",
        "You are a strict grader. Return only the requested score.",
        _TERSE_TEMPLATE, 0, 3),
    "detailed": PromptVariant(
        "detailed",
        "You are a meticulous senior security reviewer grading a junior's work.",
        _DETAILED_TEMPLATE, 0, 3),
    "scale_1_10": PromptVariant(
        "scale_1_10", POINTWISE_SYSTEM, _SCALE_1_10_TEMPLATE, 1, 10),
}


def get_pointwise_variant(name: str) -> PromptVariant:
    if name not in POINTWISE_VARIANTS:
        raise ValueError(f"unknown prompt variant {name!r}; "
                         f"choose from {list(POINTWISE_VARIANTS)}")
    return POINTWISE_VARIANTS[name]
