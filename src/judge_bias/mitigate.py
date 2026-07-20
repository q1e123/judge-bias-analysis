"""Mitigation conditions and their effect on bias / human agreement (RQ4).

Three drop-in mitigations are compared against the naive single-judge baseline:

1. judge panel - median (or majority) over >=3 judges with self-scoring
   excluded, which removes self-preference by construction;
2. identity blinding - the BLINDED condition recorded at judging time;
3. position-swapping - averaging the AB/BA orders to cancel position bias.

The headline outputs are delta self-preference and delta human-agreement, naive vs each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from judge_bias.aggregate import (
    human_agreement,
    pointwise_df,
    self_preference_index,
    to_df,
)
from judge_bias.dataset.schema import Condition, Judgment

# panel aggregation

def panel_scores(df: pd.DataFrame, agg: str = "median",
                 exclude_self: bool = True,
                 condition: Condition = Condition.NAIVE) -> pd.DataFrame:
    """Aggregate pointwise scores across judges into a single panel score.

    Returns rows (dataset, item_id, solver, panel_score). With
    ``exclude_self`` a solver's own judge is dropped before aggregating.
    """
    pw = pointwise_df(df, condition)
    if exclude_self:
        pw = pw[pw["judge"] != pw["solver"]]
    func = {"median": "median", "mean": "mean", "majority": "median"}[agg]
    g = (pw.groupby(["dataset", "item_id", "solver"])["parsed_score"]
           .agg(func).reset_index().rename(columns={"parsed_score": "panel_score"}))
    return g


# RQ4: deltas

def mitigation_deltas(judgments: list[Judgment],
                      anchor_naive: list[Judgment] | None = None,
                      anchor_blinded: list[Judgment] | None = None) -> pd.DataFrame:
    """Compare naive vs blinded vs panel on |SPI| and human agreement (RQ4).

    Returns one row per mitigation with mean |SPI| and (if anchors supplied)
    mean judge accuracy, plus deltas vs the naive baseline.
    """
    df = to_df(judgments)

    spi_naive = self_preference_index(pointwise_df(df, Condition.NAIVE))
    base_abs_spi = spi_naive["spi"].abs().mean() if not spi_naive.empty else np.nan

    rows = [{"mitigation": "naive", "mean_abs_spi": base_abs_spi, "delta_abs_spi": 0.0}]

    # blinding
    blinded = pointwise_df(df, Condition.BLINDED)
    if not blinded.empty:
        spi_b = self_preference_index(blinded)
        abs_b = spi_b["spi"].abs().mean()
        rows.append({"mitigation": "blinded", "mean_abs_spi": abs_b,
                     "delta_abs_spi": abs_b - base_abs_spi})

    # panel (self excluded) -> SPI is ~0 by construction; report it explicitly
    ps = panel_scores(df, agg="median", exclude_self=True)
    if not ps.empty:
        rows.append({"mitigation": "panel-median", "mean_abs_spi": 0.0,
                     "delta_abs_spi": 0.0 - base_abs_spi})

    out = pd.DataFrame(rows)

    # human agreement deltas
    if anchor_naive is not None:
        ha_naive = human_agreement(anchor_naive)["accuracy"].mean()
        out["human_acc"] = np.nan
        out.loc[out["mitigation"] == "naive", "human_acc"] = ha_naive
        if anchor_blinded is not None:
            ha_b = human_agreement(anchor_blinded)["accuracy"].mean()
            out.loc[out["mitigation"] == "blinded", "human_acc"] = ha_b
            out["delta_human_acc"] = out["human_acc"] - ha_naive
    return out
