from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent / "workspace_run" / "results" / "analysis.json"
IMG = HERE / "imgs"
IMG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

d = json.loads(ANALYSIS.read_text(encoding="utf-8"))

ORDER = ["llama-3.3-70b", "llama-3.1-8b", "gpt-oss-120b", "gpt-oss-20b",
         "gemma-3-27b", "gemma-3-4b", "qwen-2.5-72b", "mistral-small-24b"]
FAMILY_COLOR = {"meta": "#1f77b4", "openai": "#2ca02c", "google": "#d62728",
                "alibaba": "#9467bd", "mistral": "#ff7f0e"}
FAM = {"llama-3.3-70b": "meta", "llama-3.1-8b": "meta", "gpt-oss-120b": "openai",
       "gpt-oss-20b": "openai", "gemma-3-27b": "google", "gemma-3-4b": "google",
       "qwen-2.5-72b": "alibaba", "mistral-small-24b": "mistral"}


def save(fig, name):
    p = IMG / name
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p.name)


def _num(x):
    """None/NaN-safe float test."""
    return x is not None and x == x


def _forest(rows, pmap, xlabel, title, fname, xkey="tau"):
    """Generic horizontal forest plot with randomization-inference colouring."""
    order = sorted(rows, key=lambda m: rows[m][xkey])
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for i, m in enumerate(order):
        r = rows[m]
        pr = pmap.get(m)
        c = "#c1121f" if (_num(pr) and pr < 0.05) else "#8d99ae"
        lo, hi = r.get("ci_lo"), r.get("ci_hi")
        if _num(lo) and _num(hi):
            ax.errorbar(r[xkey], y[i], xerr=[[r[xkey] - lo], [hi - r[xkey]]],
                        fmt="o", color=c, ecolor=c, elinewidth=1.4, capsize=3, ms=5)
        else:
            ax.plot(r[xkey], y[i], "o", color=c, ms=5)
    ax.axvline(0, color="k", lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    save(fig, fname)


# Fig: per-judge leniency-adjusted self-preference tau (forest)  [spi-forest.pdf]
# Per-judge inference is not plotted: each row has only six or seven
# admissible pseudo-self cells, so the values are descriptive.
tj = {r["judge"]: r for r in d["self_by_judge"]}
_forest(tj, {}, r"leniency-adjusted self-preference $\tau_k$ (rubric points)",
        r"Per-judge self-preference (descriptive estimates)",
        "spi-forest.pdf", xkey="tau")

# Fig: seven-judge sensitivity for per-judge verbosity slopes  [verbosity-slopes.pdf]
verbosity_sensitivity = d["sensitivity_7judge"]["verbosity_ri"]
vj = {r["judge"]: {"slope": r["slope"]} for r in verbosity_sensitivity["per_judge"]}
vp = {r["judge"]: r["p_ri"] for r in verbosity_sensitivity["per_judge"]}
_forest(vj, vp, "per-judge verbosity slope (rubric points per SD of length)",
        r"Differential verbosity bias: seven-judge sensitivity",
        "verbosity-slopes.pdf", xkey="slope")

# Fig: SPI vs leniency (the confound)  [spi-vs-leniency.pdf]
spi = {r["model"]: r for r in d["spi"]}
len_by = {r["judge"]: r["leniency"] for r in d["leniency"]}
fig, ax = plt.subplots(figsize=(5.4, 3.8))
for m in ORDER:
    if m in spi and m in len_by:
        x, yv = len_by[m], spi[m]["spi"]
        ax.scatter(x, yv, s=48, color=FAMILY_COLOR[FAM[m]], zorder=3)
        ax.annotate(m, (x, yv), textcoords="offset points", xytext=(5, 3), fontsize=7)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("judge leniency (mean score minus grand mean)")
ax.set_ylabel("self-preference (SPI)")
ax.set_title("SPI is confounded by leniency")
save(fig, "spi-vs-leniency.pdf")

# Fig: family in-group preference (self excluded, multi-member families)  [family-ingroup.pdf]
fi = sorted(d["family_ingroup"], key=lambda r: r["delta"])
names = [f"{r['judge']}\n({r['family']})" for r in fi]
delta = [r["delta"] for r in fi]
fig, ax = plt.subplots(figsize=(5.4, 3.0))
ax.barh(range(len(names)), delta, color=[FAMILY_COLOR[r["family"]] for r in fi])
ax.axvline(0, color="k", lw=0.9)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel("in-family minus out-family mean score (self excluded)")
ax.set_title("Descriptive in-group preference (leniency-confounded)")
save(fig, "family-ingroup.pdf")

# Fig: size vs mean received score (self excluded, active params)  [size-bias.pdf]
sr = d["solver_received"]
xs = np.array([np.log10(r.get("active_b") or r["size_b"]) for r in sr])
ys = np.array([r["mean_score"] for r in sr])  # self-scores already excluded upstream
snames = [r["solver"] for r in sr]
r_pear = float(np.corrcoef(xs, ys)[0, 1])
# Fisher CI for the correlation (n solvers)
n = len(xs)
z = np.arctanh(r_pear)
se = 1.0 / np.sqrt(n - 3)
r_lo, r_hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
fig, ax = plt.subplots(figsize=(5.4, 3.8))
for xi, yi, nm in zip(xs, ys, snames, strict=True):
    ax.scatter(xi, yi, s=48, color=FAMILY_COLOR[FAM[nm]], zorder=3)
    ax.annotate(nm, (xi, yi), textcoords="offset points", xytext=(5, 3), fontsize=7)
ax.set_xlabel(r"solver size, $\log_{10}$ active parameters (B)")
ax.set_ylabel("mean received score (self excluded)")
ax.set_title(f"Larger solvers receive higher scores (r={r_pear:+.2f}, "
             f"95% CI [{r_lo:+.2f},{r_hi:+.2f}])")
save(fig, "size-bias.pdf")

# Fig: ranking bump chart (rankings depend on the judge)  [ranking-bump.pdf]
rbj = d["ranking_by_judge"]
judges = [j for j in ORDER if j in rbj]
rank = {j: {s: i + 1 for i, s in enumerate(rbj[j])} for j in judges}
fig, ax = plt.subplots(figsize=(6.4, 4.0))
x = np.arange(len(judges))
for s in ORDER:
    ys = [rank[j].get(s, np.nan) for j in judges]
    ax.plot(x, ys, "-o", color=FAMILY_COLOR[FAM[s]], lw=1.4, ms=4, zorder=2)
    if not np.isnan(ys[-1]):
        ax.annotate(s, (x[-1], ys[-1]), textcoords="offset points",
                    xytext=(6, 0), fontsize=6.5, va="center", color=FAMILY_COLOR[FAM[s]])
ax.set_xticks(x)
ax.set_xticklabels(judges, rotation=45, ha="right")
ax.set_yticks(range(1, len(ORDER) + 1))
ax.set_ylabel("solver rank under judge (1 = best)")
ax.set_xlim(-0.3, len(judges) - 1 + 1.7)
ax.invert_yaxis()
ax.set_title("Rankings depend on the judge (Bradley-Terry)")
save(fig, "ranking-bump.pdf")

# Fig: directional first-position win rate (position bias has a direction)  [position-bias.pdf]
pd_ = sorted(d["position_direction"]["per_judge"], key=lambda r: r["first_win_rate"])
pnames = [r["judge"] for r in pd_]
dev_a = [r["first_win_rate"] - 0.5 for r in pd_]
sigcol = ["#c1121f" if (_num(r["p_holm"]) and r["p_holm"] < 0.05) else "#8d99ae"
          for r in pd_]
fig, ax = plt.subplots(figsize=(5.4, 3.0))
ax.barh(range(len(pnames)), dev_a, color=sigcol)
ax.axvline(0, color="k", lw=0.9)
ax.set_yticks(range(len(pnames)))
ax.set_yticklabels(pnames)
ax.set_xlabel("first-position (slot A) win rate minus 0.5")
ax.set_title("Position bias has a direction and splits across judges")
save(fig, "position-bias.pdf")

# Fig: judge x solver bias heatmap (raw mean minus per-solver consensus)  [bias-heatmap.pdf]
bm = d["bias_matrix_raw"]
M = np.full((len(ORDER), len(ORDER)), np.nan)
for i, jd in enumerate(ORDER):
    row = bm.get(jd, {})
    for k, sv in enumerate(ORDER):
        if sv in row and row[sv] is not None:
            M[i, k] = row[sv]
consensus = np.nanmean(M, axis=0, keepdims=True)
dev = M - consensus
fig, ax = plt.subplots(figsize=(5.6, 4.6))
vmax = np.nanmax(np.abs(dev))
im = ax.imshow(dev, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels(ORDER, rotation=45, ha="right")
ax.set_yticks(range(len(ORDER)))
ax.set_yticklabels(ORDER)
ax.set_xlabel("solver (answer author)")
ax.set_ylabel("judge")
for i in range(len(ORDER)):
    ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, edgecolor="k", lw=1.2))
    for k in range(len(ORDER)):
        if not np.isnan(dev[i, k]):
            ax.text(k, i, f"{dev[i, k]:+.2f}", ha="center", va="center", fontsize=5.5, color="k")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("score minus per-solver consensus")
ax.set_title("Judge-by-solver bias (boxed diagonal = self)")
save(fig, "bias-heatmap.pdf")

# Fig: anchor accuracy vs adjusted self-preference tau  [anchor-vs-spi.pdf]
tau = {r["judge"]: r["tau"] for r in d["self_by_judge"]}
acc = {r["judge"]: r["accuracy"] for r in d["human_agreement"]}
js = [j for j in tau if j in acc]
ax_x = np.array([acc[j] for j in js])
ax_y = np.array([tau[j] for j in js])
r = float(np.corrcoef(ax_x, ax_y)[0, 1])
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for j in js:
    ax.scatter(acc[j], tau[j], s=48, color=FAMILY_COLOR[FAM[j]], zorder=3)
    ax.annotate(j, (acc[j], tau[j]), textcoords="offset points", xytext=(5, 3), fontsize=7)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("objective anchor accuracy (judge vs. WMDP truth)")
ax.set_ylabel(r"leniency-adjusted self-preference $\tau_k$")
ax.set_title(f"Accuracy--self-preference association is uncertain (r={r:+.2f})")
save(fig, "anchor-vs-spi.pdf")

print("done ->", IMG)
