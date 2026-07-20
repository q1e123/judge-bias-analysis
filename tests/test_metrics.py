"""Metric/estimator tests on directly-constructed judgments (no model calls).

Synthesise a small judgment set with a known structure (a size effect, a
same-family effect, and per-judge self-preference) and check that the analysis
recovers those signs.
"""

from __future__ import annotations

import math

import pytest

from judge_bias import aggregate, inference, stats
from judge_bias.config import ModelSpec, RunConfig
from judge_bias.dataset.schema import Condition, JudgeMode, Judgment

# 4 models: two families, two sizes each -> size and family are both identifiable.
MODELS = [
    ModelSpec(name="a-small", provider="openai_compat", model_id="a", family="meta", size_b=8.0),
    ModelSpec(name="a-big", provider="openai_compat", model_id="b", family="meta", size_b=70.0),
    ModelSpec(name="b-small", provider="openai_compat", model_id="c", family="mistral", size_b=20.0),
    ModelSpec(name="b-big", provider="openai_compat", model_id="d", family="mistral", size_b=120.0),
]
NAMES = [m.name for m in MODELS]
SIZE = {m.name: m.size_b for m in MODELS}
FAM = {m.name: m.family for m in MODELS}
N_ITEMS = 15


def _score(judge, solver, i):
    """Continuous rubric score with a built-in size, family, and self effect."""
    val = (1.0
           + 0.5 * math.log10(SIZE[solver])          # size effect  -> log10_size > 0
           + 0.5 * (FAM[judge] == FAM[solver])        # family effect -> same_family > 0
           + 0.7 * (judge == solver)                  # self effect  -> is_self > 0
           + 0.15 * ((i % 3) - 1))                    # item variation (absorbed by FE)
    return max(0.0, min(3.0, val))


@pytest.fixture(scope="module")
def bundle():
    judgments: list[Judgment] = []
    for i in range(N_ITEMS):
        iid = f"i{i:02d}"
        for j in NAMES:
            for m in NAMES:
                judgments.append(Judgment(
                    dataset="cyberseceval", item_id=iid, judge=j,
                    mode=JudgeMode.POINTWISE, condition=Condition.NAIVE, solver=m,
                    parsed_score=_score(j, m, i),
                    answer_len=40 + 5 * NAMES.index(m) + (i % 4)))
        # a little pairwise so the pairwise-dependent metrics populate
        for j in NAMES:
            for a, b in (("a-small", "a-big"), ("b-small", "b-big")):
                win = a if SIZE[a] > SIZE[b] else b
                judgments.append(Judgment(
                    dataset="cyberseceval", item_id=iid, judge=j,
                    mode=JudgeMode.PAIRWISE, condition=Condition.NAIVE,
                    solver_a=a, solver_b=b, order="AB", winner=win))
                judgments.append(Judgment(
                    dataset="cyberseceval", item_id=iid, judge=j,
                    mode=JudgeMode.PAIRWISE, condition=Condition.SWAPPED,
                    solver_a=a, solver_b=b, order="BA", winner=win))
    return RunConfig(models=MODELS), judgments


def test_spi_recovers_self_preference(bundle):
    _, judgments = bundle
    res = stats.spi_with_ci(judgments, n_boot=300)
    assert {"dz", "win_rate", "spi"}.issubset(res.columns)
    # every judge was given a +0.7 self bump, so own > other on essentially every item
    assert res["win_rate"].mean() > 0.5
    assert (res["spi"] > 0).all()


def test_self_preference_regression_isolates_self(bundle):
    cfg, judgments = bundle
    reg = stats.self_preference_regression(judgments, cfg, cluster=True)
    assert reg.params["is_self"] > 0
    assert "cluster" in reg.cov_type
    assert any(t.startswith("C(item_id)") for t in reg.params.index)


def test_size_and_family_regression_signs(bundle):
    cfg, judgments = bundle
    reg = stats.size_family_regression(judgments, cfg, cluster=True)
    assert reg.params["log10_size"] > 0
    assert reg.params["same_family"] > 0


def test_krippendorff_alpha_returns_number(bundle):
    _, judgments = bundle
    a = stats.krippendorff_alpha(judgments)
    assert isinstance(a, float)  # NaN is acceptable; must not raise
def test_summarise_produces_core_metrics(bundle):
    cfg, judgments = bundle
    summary = aggregate.summarise(judgments, cfg)
    for key in ("bias_matrix", "leniency", "scale_usage", "position_bias"):
        assert key in summary


def test_exact_sign_flip_has_no_monte_carlo_smoothing():
    vals = [0.1] * 8
    assert inference._sign_flip_p(vals, 0.1) == pytest.approx(2 / 256)


def test_active_parameter_proxy_is_explicit():
    spec = ModelSpec(name="moe", provider="mock", model_id="moe", family="x",
                     size_b=120.0, active_size_b=5.1)
    assert spec.size_for_regression("total") == 120.0
    assert spec.size_for_regression("active") == 5.1


def test_anchor_ties_count_as_errors():
    anchor = [
        Judgment(dataset="anchor", item_id="a", judge="j", mode=JudgeMode.PAIRWISE,
                 condition=Condition.NAIVE, solver_a="__chosen__", solver_b="__rejected__",
                 winner="__chosen__"),
        Judgment(dataset="anchor", item_id="b", judge="j", mode=JudgeMode.PAIRWISE,
                 condition=Condition.NAIVE, solver_a="__chosen__", solver_b="__rejected__",
                 winner="tie"),
    ]
    row = aggregate.human_agreement(anchor).iloc[0]
    assert row["accuracy"] == pytest.approx(0.5)
    assert row["n"] == 2
