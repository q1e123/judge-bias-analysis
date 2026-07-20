from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from judge_bias.config import RunConfig
from judge_bias.dataset.schema import Condition, JudgeMode, Judgment


def to_df(judgments: list[Judgment]) -> pd.DataFrame:
    return pd.DataFrame(j.model_dump() for j in judgments)


def pointwise_df(df: pd.DataFrame, condition: Condition = Condition.NAIVE,
                 exclude_refused: bool = False,
                 variant: str | None = "default") -> pd.DataFrame:
    """Pointwise judgments for a condition. ``variant`` selects the judge-prompt
    template ("default" for the main analysis, None to keep every variant)."""
    cond = condition.value if isinstance(condition, Condition) else condition
    out = df[(df["mode"] == JudgeMode.POINTWISE.value) & (df["condition"] == cond)]
    out = out[out["parsed_score"].notna()]
    if variant is not None and "prompt_variant" in out.columns:
        out = out[out["prompt_variant"] == variant]
    if exclude_refused and "solver_refused" in out.columns:
        out = out[~out["solver_refused"].astype(bool)]
    return out.copy()


def pairwise_df(df: pd.DataFrame, condition: Condition = Condition.NAIVE) -> pd.DataFrame:
    cond = condition.value if isinstance(condition, Condition) else condition
    out = df[(df["mode"] == JudgeMode.PAIRWISE.value) & (df["condition"] == cond)]
    return out[out["winner"].notna()].copy()

def self_preference_index(pw: pd.DataFrame) -> pd.DataFrame:
    """SPI(m) = mean(score m gives m's own answers) - mean(score others give m).

    Returns one row per model with the two component means and the index.
    """
    rows = []
    models = sorted(set(pw["solver"]) | set(pw["judge"]))
    for m in models:
        own = pw[(pw["judge"] == m) & (pw["solver"] == m)]["parsed_score"]
        others = pw[(pw["judge"] != m) & (pw["solver"] == m)]["parsed_score"]
        if len(own) == 0 or len(others) == 0:
            continue
        rows.append({
            "model": m,
            "self_mean": own.mean(),
            "other_mean": others.mean(),
            "spi": own.mean() - others.mean(),
            "n_self": len(own),
            "n_other": len(others),
        })
    cols = ["model", "self_mean", "other_mean", "spi", "n_self", "n_other"]
    if not rows:  # partial/empty judgment set - don't crash, return an empty frame
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("spi", ascending=False).reset_index(drop=True)

def bias_matrix(pw: pd.DataFrame, row_normalise: bool = True) -> pd.DataFrame:
    """B[judge, solver] = mean pointwise score. Row-normalised exposes inflation."""
    mat = pw.pivot_table(index="judge", columns="solver", values="parsed_score",
                         aggfunc="mean")
    if row_normalise:
        mat = mat.sub(mat.mean(axis=1), axis=0)
    return mat

def _bt_from_wins(wins: np.ndarray, iters: int = 200, tol: float = 1e-9) -> np.ndarray:
    """Zermelo/MM iteration for Bradley-Terry strengths (normalised to mean 1)."""
    n = wins.shape[0]
    p = np.ones(n)
    total = wins + wins.T
    w_i = wins.sum(axis=1)  # total wins per player
    for _ in range(iters):
        p_old = p.copy()
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                nij = total[i, j]
                if nij > 0:
                    denom += nij / (p[i] + p[j])
            if denom > 0 and w_i[i] > 0:
                p[i] = w_i[i] / denom
        p = p / p.mean()
        if np.max(np.abs(p - p_old)) < tol:
            break
    return p


def bradley_terry_by_judge(pair: pd.DataFrame) -> pd.DataFrame:
    """Per-judge Bradley-Terry strength for each solver (ties = half a win)."""
    solvers = sorted(set(pair["solver_a"]) | set(pair["solver_b"]))
    idx = {s: i for i, s in enumerate(solvers)}
    rows = []
    for judge, g in pair.groupby("judge"):
        wins = np.zeros((len(solvers), len(solvers)))
        for _, r in g.iterrows():
            a, b = r["solver_a"], r["solver_b"]
            if r["winner"] == "tie":
                wins[idx[a], idx[b]] += 0.5
                wins[idx[b], idx[a]] += 0.5
            elif r["winner"] == a:
                wins[idx[a], idx[b]] += 1
            elif r["winner"] == b:
                wins[idx[b], idx[a]] += 1
        strengths = _bt_from_wins(wins)
        for s in solvers:
            rows.append({"judge": judge, "solver": s, "strength": strengths[idx[s]]})
    return pd.DataFrame(rows)


def judge_leaderboards(bt: pd.DataFrame) -> dict[str, list[str]]:
    """Per-judge ranking of solvers (best first) from Bradley-Terry strengths."""
    out = {}
    for judge, g in bt.groupby("judge"):
        out[judge] = list(g.sort_values("strength", ascending=False)["solver"])
    return out


def ranking_volatility(bt: pd.DataFrame) -> dict:
    """Pairwise Kendall's τ between judges' leaderboards + top-1 churn (RQ3)."""
    from scipy.stats import kendalltau

    piv = bt.pivot(index="solver", columns="judge", values="strength")
    judges = list(piv.columns)
    taus = {}
    for j1, j2 in itertools.combinations(judges, 2):
        tau, _ = kendalltau(piv[j1], piv[j2])
        taus[(j1, j2)] = tau
    leaderboards = judge_leaderboards(bt)
    top1 = {j: lb[0] for j, lb in leaderboards.items()}
    n_distinct_top1 = len(set(top1.values()))
    return {
        "kendall_tau": taus,
        "mean_tau": float(np.nanmean(list(taus.values()))) if taus else float("nan"),
        "top1_by_judge": top1,
        "n_distinct_top1": n_distinct_top1,
        "leaderboards": leaderboards,
    }


# control: position-bias rate

def position_bias_rate(df: pd.DataFrame) -> pd.DataFrame:
    """% of pairs whose winner flips between order AB (NAIVE) and BA (SWAPPED)."""
    ab = pairwise_df(df, Condition.NAIVE)
    ba = pairwise_df(df, Condition.SWAPPED)
    keys = ["dataset", "item_id", "judge", "solver_a", "solver_b"]
    merged = ab.merge(ba, on=keys, suffixes=("_ab", "_ba"))
    rows = []
    for judge, g in merged.groupby("judge"):
        decided = g[(g["winner_ab"] != "tie") & (g["winner_ba"] != "tie")]
        flips = (decided["winner_ab"] != decided["winner_ba"]).sum()
        n = len(decided)
        rows.append({"judge": judge, "n_pairs": n,
                     "flip_rate": flips / n if n else float("nan")})
    return pd.DataFrame(rows).sort_values("flip_rate", ascending=False).reset_index(drop=True)


# verbosity confound

def verbosity_correlation(pw: pd.DataFrame) -> pd.DataFrame:
    """Per-judge Spearman correlation between answer length and score (RQ2)."""
    from scipy.stats import spearmanr

    rows = []
    for judge, g in pw.groupby("judge"):
        g = g[g["answer_len"].notna()]
        if len(g) < 3:
            continue
        rho, p = spearmanr(g["answer_len"], g["parsed_score"])
        rows.append({"judge": judge, "spearman_len_score": rho, "p": p, "n": len(g)})
    return pd.DataFrame(rows)


# judge leniency / severity + scale calibration (RQ2 / data-quality)

def judge_leniency(pw: pd.DataFrame) -> pd.DataFrame:
    """Per-judge mean score minus the grand mean - severity (<0) / leniency (>0)."""
    if pw.empty:
        return pd.DataFrame(columns=["judge", "mean_score", "leniency", "n"])
    grand = pw["parsed_score"].mean()
    rows = [{"judge": j, "mean_score": g["parsed_score"].mean(),
             "leniency": g["parsed_score"].mean() - grand, "n": len(g)}
            for j, g in pw.groupby("judge")]
    return pd.DataFrame(rows).sort_values("leniency", ascending=False).reset_index(drop=True)


def scale_usage(pw: pd.DataFrame) -> pd.DataFrame:
    """Per-judge use of the rating scale: Shannon entropy + distinct values + std.

    A judge that collapses to one value (entropy ~ 0) is uninformative regardless
    of bias - a calibration / data-quality flag worth reporting alongside SPI."""
    if pw.empty:
        return pd.DataFrame(columns=["judge", "distinct", "entropy_bits", "std", "n"])
    rows = []
    for judge, g in pw.groupby("judge"):
        s = g["parsed_score"].dropna()
        p = s.value_counts(normalize=True)
        ent = float(-(p * np.log2(p)).sum()) if len(p) else 0.0
        rows.append({"judge": judge, "distinct": int(s.nunique()),
                     "entropy_bits": ent, "std": float(s.std(ddof=0)), "n": len(s)})
    return pd.DataFrame(rows).sort_values("entropy_bits").reset_index(drop=True)


def family_ingroup_bias(pw: pd.DataFrame, specs) -> pd.DataFrame:
    """Per-judge mean score to SAME-family solvers vs OTHER-family (self excluded).

    Isolates in-group family favouritism as an effect size, separate from the
    regression's `same_family` term and from self-preference (RQ2)."""
    cols = ["judge", "family", "in_family", "out_family", "delta", "n_in", "n_out"]
    fam = {s.name: s.family for s in specs} if specs else {}
    if pw.empty or not fam:
        return pd.DataFrame(columns=cols)
    rows = []
    for judge, g in pw.groupby("judge"):
        jf = fam.get(judge)
        g = g[g["solver"] != judge]  # exclude self so this isn't SPI
        same = g[g["solver"].map(fam) == jf]["parsed_score"]
        diff = g[g["solver"].map(fam) != jf]["parsed_score"]
        if not len(same) or not len(diff):
            continue
        rows.append({"judge": judge, "family": jf,
                     "in_family": same.mean(), "out_family": diff.mean(),
                     "delta": same.mean() - diff.mean(),
                     "n_in": len(same), "n_out": len(diff)})
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)


# cross-cutting: human agreement

def human_agreement(anchor: list[Judgment]) -> pd.DataFrame:
    """Judge accuracy on human chosen/rejected anchor pairs (κ-style agreement).

    Anchor judgments are pairwise records with solver_a='__chosen__',
    solver_b='__rejected__'. A correct judgment picks '__chosen__'.
    """
    df = to_df(anchor) if anchor and isinstance(anchor[0], Judgment) else pd.DataFrame(anchor)
    if df.empty:
        return pd.DataFrame(columns=["judge", "accuracy", "acc_lo", "acc_hi",
                                     "n", "tie_rate"])
    df = df[df["winner"].notna()]
    rows = []
    for judge, g in df.groupby("judge"):
        correct = int((g["winner"] == "__chosen__").sum())
        n = len(g)
        acc = correct / n if n else float("nan")
        lo, hi = _wilson_ci(correct, n)
        # tie/abstention rate on the anchor: a judge that hedges on hazardous
        # items scores as inaccurate for safety reasons, not incompetence
        tie_rate = float((g["winner"] == "tie").mean()) if len(g) else float("nan")
        rows.append({"judge": judge, "accuracy": acc, "acc_lo": lo, "acc_hi": hi,
                     "n": n, "tie_rate": tie_rate})
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return centre - half, centre + half


# refusal confound

# Security (cyber) datasets vs. general-domain controls. Keyed by the
# TaskItem.dataset value. Used for the refusal split, the domain contrast, and
# the security-only per-dataset report.
SECURITY_DATASETS = {
    "cyberseceval", "synthetic-security", "synthetic-offensive",
    "synthetic-defensive", "cti-bench", "wmdp",
}
GENERAL_DATASETS = {"mt-bench"}


def refusal_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Per-solver refusal rate and the score gap a judge assigns to refusals.

    A judge that conflates "safely refused" with "low quality" inflates the
    apparent quality gap between safety-leaning and compliant models. Reporting
    this lets the paper show bias on the *non-refused* subset separately.
    """
    pw = pointwise_df(df, Condition.NAIVE)
    if "solver_refused" not in pw.columns:
        return pd.DataFrame(columns=["solver", "refusal_rate", "score_refused",
                                     "score_nonrefused", "score_gap"])
    rows = []
    for solver, g in pw.groupby("solver"):
        ref = g[g["solver_refused"].astype(bool)]["parsed_score"]
        non = g[~g["solver_refused"].astype(bool)]["parsed_score"]
        rows.append({
            "solver": solver,
            "refusal_rate": len(ref) / len(g) if len(g) else float("nan"),
            "score_refused": ref.mean() if len(ref) else float("nan"),
            "score_nonrefused": non.mean() if len(non) else float("nan"),
            "score_gap": (non.mean() - ref.mean()) if len(ref) and len(non) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("refusal_rate", ascending=False).reset_index(drop=True)


def domain_label(dataset: str) -> str:
    if dataset in SECURITY_DATASETS:
        return "security"
    if dataset in GENERAL_DATASETS:
        return "general"
    return "other"


def domain_contrast(judgments: list[Judgment], n_boot: int = 5000,
                    seed: int = 7) -> dict:
    """Is self-preference larger in security
    than in general evaluation?

    Computes mean self-preference within each domain and a bootstrap CI on the
    security − general difference. Returns the per-domain means and the delta.
    """
    df = to_df(judgments)
    pw = pointwise_df(df, Condition.NAIVE)
    if pw.empty:
        return {}
    pw = pw.assign(domain=pw["dataset"].map(domain_label))

    def domain_spi_samples(sub: pd.DataFrame) -> np.ndarray:
        """Per-model SPI (self − other) within a domain, on identical answers."""
        vals = []
        for m in sorted(set(sub["solver"]) | set(sub["judge"])):
            own = sub[(sub["judge"] == m) & (sub["solver"] == m)]["parsed_score"]
            oth = sub[(sub["judge"] != m) & (sub["solver"] == m)]["parsed_score"]
            if len(own) and len(oth):
                vals.append(own.mean() - oth.mean())
        return np.asarray(vals, dtype=float)

    out = {}
    per_domain = {}
    for dom in ("security", "general"):
        s = domain_spi_samples(pw[pw["domain"] == dom])
        if len(s):
            per_domain[dom] = {"mean_spi": float(np.mean(s)), "n_models": int(len(s))}
    out["per_domain"] = per_domain
    if {"security", "general"} <= set(per_domain):
        sec = domain_spi_samples(pw[pw["domain"] == "security"])
        gen = domain_spi_samples(pw[pw["domain"] == "general"])
        rng = np.random.default_rng(seed)
        boots = np.array([
            rng.choice(sec, len(sec), replace=True).mean()
            - rng.choice(gen, len(gen), replace=True).mean()
            for _ in range(n_boot)
        ])
        out["security_minus_general"] = float(sec.mean() - gen.mean())
        out["ci_lo"], out["ci_hi"] = (float(np.percentile(boots, 2.5)),
                                      float(np.percentile(boots, 97.5)))
    return out


# robustness across judge prompts

def robustness_summary(judgments: list[Judgment]) -> pd.DataFrame:
    """Mean self-preference per judge-prompt variant (robustness check).

    A top-tier LLM-judge paper must show the bias survives paraphrase and
    re-scaling of the judge prompt; a stable SPI across rows is that evidence.
    """
    df = to_df(judgments)
    variants = sorted(df.loc[df["mode"] == JudgeMode.POINTWISE.value,
                             "prompt_variant"].dropna().unique()) \
        if "prompt_variant" in df.columns else ["default"]
    rows = []
    for v in variants:
        pw = pointwise_df(df, Condition.NAIVE, variant=v)
        spi = self_preference_index(pw)
        if spi.empty:
            continue
        rows.append({
            "prompt_variant": v,
            "mean_spi": spi["spi"].mean(),
            "mean_abs_spi": spi["spi"].abs().mean(),
            "n_models": len(spi),
        })
    return pd.DataFrame(rows)


# convergent validity: pointwise vs pairwise

def convergent_validity(judgments: list[Judgment]) -> dict:
    """Do pointwise and pairwise judging agree on the ranking? (construct validity)

    Per judge, Spearman correlation between mean pointwise score per solver and
    Bradley-Terry strength per solver; plus the pooled correlation. High values
    mean the two protocols measure the same thing despite different bias surfaces.
    """
    from scipy.stats import spearmanr

    df = to_df(judgments)
    pw = pointwise_df(df, Condition.NAIVE)
    pair = pairwise_df(df, Condition.NAIVE)
    if pw.empty or pair.empty:
        return {}
    bt = bradley_terry_by_judge(pair)
    rows = []
    for judge in sorted(set(pw["judge"]) & set(bt["judge"])):
        p = pw[pw["judge"] == judge].groupby("solver")["parsed_score"].mean()
        b = bt[bt["judge"] == judge].set_index("solver")["strength"]
        common = p.index.intersection(b.index)
        if len(common) >= 3:
            rho, _ = spearmanr(p.loc[common], b.loc[common])
            rows.append({"judge": judge, "spearman_pw_vs_pair": rho, "n": len(common)})
    per_judge = pd.DataFrame(rows)
    # pooled: solver mean pointwise vs panel BT strength
    pooled_p = pw.groupby("solver")["parsed_score"].mean()
    pooled_b = bt.groupby("solver")["strength"].mean()
    common = pooled_p.index.intersection(pooled_b.index)
    pooled_rho = float(spearmanr(pooled_p.loc[common], pooled_b.loc[common])[0]) \
        if len(common) >= 3 else float("nan")
    return {"per_judge": per_judge, "pooled_spearman": pooled_rho,
            "mean_spearman": float(per_judge["spearman_pw_vs_pair"].mean())
            if not per_judge.empty else float("nan")}


# convenience bundle

def summarise(judgments: list[Judgment], cfg: RunConfig | None = None,
              anchor: list[Judgment] | None = None) -> dict:
    """Compute the headline metrics in one call (used by the CLI / report)."""
    df = to_df(judgments)
    pw = pointwise_df(df, Condition.NAIVE)
    result = {
        "spi": self_preference_index(pw),
        "bias_matrix": bias_matrix(pw),
        "verbosity": verbosity_correlation(pw),
        "leniency": judge_leniency(pw),
        "scale_usage": scale_usage(pw),
    }
    if cfg is not None:
        result["family_ingroup"] = family_ingroup_bias(pw, cfg.models)
    pair = pairwise_df(df, Condition.NAIVE)
    if not pair.empty:
        bt = bradley_terry_by_judge(pair)
        result["bradley_terry"] = bt
        result["ranking_volatility"] = ranking_volatility(bt)
        result["position_bias"] = position_bias_rate(df)
    if anchor:
        result["human_agreement"] = human_agreement(anchor)
    return result
