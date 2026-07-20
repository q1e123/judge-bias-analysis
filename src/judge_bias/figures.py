"""Headline figures: bias-matrix heatmap, SPI forest plot, ranking-flip
diagram. All save to PNG and return the matplotlib Figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / reproducible
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _save(fig, path: str | Path | None):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def bias_heatmap(matrix: pd.DataFrame, path: str | Path | None = None):
    """Row-normalised judge/solver bias matrix - the headline heatmap."""
    fig, ax = plt.subplots(figsize=(1.1 * matrix.shape[1] + 2, 0.7 * matrix.shape[0] + 2))
    vmax = np.nanmax(np.abs(matrix.to_numpy()))
    im = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("solver (answer author)")
    ax.set_ylabel("judge")
    ax.set_title("Bias matrix (row-normalised mean score)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.to_numpy()[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        color="black" if abs(v) < vmax * 0.6 else "white")
    fig.colorbar(im, ax=ax, label="score − judge row mean")
    fig.tight_layout()
    return _save(fig, path)


def spi_forest(spi_df: pd.DataFrame, path: str | Path | None = None):
    """Forest plot of per-model SPI with 95% CIs (expects ci_lo/ci_hi cols)."""
    df = spi_df.sort_values("spi")
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(df) + 1.5))
    y = np.arange(len(df))
    has_ci = {"ci_lo", "ci_hi"}.issubset(df.columns)
    if has_ci:
        err = np.vstack([df["spi"] - df["ci_lo"], df["ci_hi"] - df["spi"]])
        ax.errorbar(df["spi"], y, xerr=err, fmt="o", color="#34568B", capsize=3)
    else:
        ax.plot(df["spi"], y, "o", color="#34568B")
    ax.axvline(0, color="grey", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("Self-Preference Index  (self − others)")
    ax.set_title("Self-preference by model (95% bootstrap CI)")
    fig.tight_layout()
    return _save(fig, path)


def ranking_flip(leaderboards: dict[str, list[str]], path: str | Path | None = None):
    """Bump chart: each solver's rank under each judge - exposes top-1 churn."""
    judges = list(leaderboards)
    solvers = sorted({s for lb in leaderboards.values() for s in lb})
    rank = {j: {s: i + 1 for i, s in enumerate(lb)} for j, lb in leaderboards.items()}
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(1.4 * len(judges) + 2, 0.5 * len(solvers) + 2))
    for k, s in enumerate(solvers):
        ys = [rank[j].get(s, np.nan) for j in judges]
        ax.plot(range(len(judges)), ys, "-o", color=cmap(k % 10), label=s)
    ax.set_xticks(range(len(judges)))
    ax.set_xticklabels(judges, rotation=45, ha="right")
    ax.set_ylabel("rank (1 = best)")
    ax.invert_yaxis()
    ax.set_title("Judge-dependent solver ranking (ranking-flip diagram)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def render_all(summary: dict, outdir: str | Path) -> list[Path]:
    """Render the headline figure set from :func:`aggregate.summarise` output."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    if "bias_matrix" in summary:
        bias_heatmap(summary["bias_matrix"], outdir / "bias_matrix.png")
        paths.append(outdir / "bias_matrix.png")
    if "spi" in summary and not summary["spi"].empty:
        spi_forest(summary["spi"], outdir / "spi_forest.png")
        paths.append(outdir / "spi_forest.png")
    if "ranking_volatility" in summary:
        ranking_flip(summary["ranking_volatility"]["leaderboards"],
                     outdir / "ranking_flip.png")
        paths.append(outdir / "ranking_flip.png")
    plt.close("all")
    return paths
