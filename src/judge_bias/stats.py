"""Statistical analysis: bootstrap CIs, paired tests, the size/family
regression, Holm correction, Krippendorff's α.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from judge_bias.aggregate import pointwise_df, to_df
from judge_bias.config import RunConfig
from judge_bias.dataset.schema import Condition


def spi_with_ci(judgments, n_boot: int = 5000, seed: int = 7) -> pd.DataFrame:
    """Per-model SPI with a bootstrap CI on the (self − other) difference and a
    paired Wilcoxon test, Holm-corrected across models (RQ1).
    """
    from scipy.stats import wilcoxon

    df = to_df(judgments)
    pw = pointwise_df(df, Condition.NAIVE)
    models = sorted(set(pw["solver"]) | set(pw["judge"]))
    rows = []
    pvals = []
    for m in models:
        own = pw[(pw["judge"] == m) & (pw["solver"] == m)]
        others = pw[(pw["judge"] != m) & (pw["solver"] == m)]
        if len(own) == 0 or len(others) == 0:
            continue
        # bootstrap CI on difference of means
        rng = np.random.default_rng(seed)
        o, oo = own["parsed_score"].to_numpy(), others["parsed_score"].to_numpy()
        diffs = np.array([
            rng.choice(o, len(o), replace=True).mean()
            - rng.choice(oo, len(oo), replace=True).mean()
            for _ in range(n_boot)
        ])
        spi = o.mean() - oo.mean()
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        # paired Wilcoxon per item: own vs mean-other on shared items
        merged = own.groupby("item_id")["parsed_score"].mean().to_frame("own").join(
            others.groupby("item_id")["parsed_score"].mean().to_frame("other"), how="inner")
        p = np.nan
        d = merged["own"] - merged["other"]
        if len(merged) >= 3 and d.abs().sum() > 0:
            try:
                _, p = wilcoxon(merged["own"], merged["other"])
            except ValueError:
                p = np.nan
        # effect sizes: matched-pairs Cohen's dz and the own>other win rate
        dz = float(d.mean() / d.std(ddof=1)) if len(d) > 1 and d.std(ddof=1) > 0 else np.nan
        win_rate = float((d > 0).mean()) if len(d) else np.nan
        rows.append({"model": m, "spi": spi, "ci_lo": lo, "ci_hi": hi,
                     "dz": dz, "win_rate": win_rate,
                     "n_items": len(merged), "p_raw": p})
        pvals.append(p)

    res = pd.DataFrame(rows)
    if not res.empty:
        res["p_holm"] = holm_correction(res["p_raw"].to_numpy())
    return res.sort_values("spi", ascending=False).reset_index(drop=True)


# multiple comparison

def holm_correction(pvals) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (NaNs passed through)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * p[i])
        adj = max(adj, prev)  # enforce monotonicity
        out[i] = adj
        prev = adj
    return out


# size / family / self regression (RQ1+RQ2)

def _prep_regression_df(judgments, cfg: RunConfig,
                        anchor_scores: dict | None = None,
                        exclude_refused: bool = False,
                        condition: Condition = Condition.NAIVE) -> pd.DataFrame:
    """Build the tidy frame for the fixed-effects judgment regressions.

    Adds the design columns shared by the size/family and self-preference
    models: ``is_self`` (judge==solver), ``same_family``, total- and
    active-parameter size proxies, ``verbosity``, and optional per-model
    ``correctness``. ``condition`` selects the naive or blinded pointwise pass.
    """
    df = to_df(judgments)
    pw = pointwise_df(df, condition, exclude_refused=exclude_refused).copy()
    spec_by_name = {m.name: m for m in cfg.models}

    def size_of(name, measure):
        s = spec_by_name.get(name)
        v = s.size_for_regression(measure) if s else None
        return np.log10(v) if v else np.nan

    def family_of(name):
        s = spec_by_name.get(name)
        return s.family if s else "unknown"

    pw["is_self"] = (pw["judge"] == pw["solver"]).astype(int)
    pw["log10_size"] = pw["solver"].map(lambda name: size_of(name, "total"))
    pw["log10_active_size"] = pw["solver"].map(lambda name: size_of(name, "active"))
    pw["solver_family"] = pw["solver"].map(family_of)
    pw["judge_family"] = pw["judge"].map(family_of)
    pw["same_family"] = (pw["solver_family"] == pw["judge_family"]).astype(int)
    # same family but a different model, so the self and family terms are disjoint
    # and a self indicator is not double-counted by the family term
    pw["same_family_nonself"] = ((pw["solver_family"] == pw["judge_family"])
                                 & (pw["judge"] != pw["solver"])).astype(int)
    pw["verbosity"] = pw["answer_len"].astype(float)
    # one level per exact solver answer, for the answer-level fixed-effects variant
    pw["answer"] = pw["solver"] + "|" + pw["item_id"]
    if anchor_scores:
        pw["correctness"] = pw["solver"].map(lambda s: anchor_scores.get(s, np.nan))
    return pw.dropna(subset=["log10_size", "parsed_score", "verbosity"])


def _fit_clustered(formula: str, data: pd.DataFrame, cluster: bool):
    """OLS fit. With ``cluster`` use two-way cluster-robust SEs (item & judge),
    since scores are correlated within both the same item and the same judge -
    plain OLS SEs would be anticonservative.
    """
    import statsmodels.formula.api as smf

    model = smf.ols(formula, data=data)
    if cluster:
        # statsmodels multiway clustering needs integer group codes, not strings
        groups = np.column_stack([
            pd.factorize(data["item_id"])[0],
            pd.factorize(data["judge"])[0],
        ])
        return model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return model.fit()


def self_preference_regression(judgments, cfg: RunConfig,
                               anchor_scores: dict | None = None,
                               cluster: bool = True,
                               exclude_refused: bool = False) -> object:
    """Canonical self-preference estimator (RQ1) - identifies bias net of
    quality and judge leniency via two-way fixed effects:

        score ~ is_self + same_family_nonself + C(judge) + C(solver) + C(item)

    ``C(item)`` absorbs item difficulty, ``C(judge)`` absorbs judge leniency, and
    ``same_family_nonself`` is disjoint from ``is_self`` (it fires only on
    different-model same-family pairs), so the is_self coefficient is the clean
    self-preference effect net of any same-family favouritism. Using the
    self-inclusive ``same_family`` here would let the family term absorb the self
    effect and drive is_self spuriously to zero.
    """
    pw = _prep_regression_df(judgments, cfg, anchor_scores, exclude_refused)
    formula = ("parsed_score ~ is_self + same_family_nonself "
               "+ C(judge) + C(solver) + C(item_id)")
    if anchor_scores:
        formula += " + correctness"
    return _fit_clustered(formula, pw, cluster)


def _san(name: str) -> str:
    """patsy-safe column name for a model id (only alnum + underscore)."""
    return "sf_" + "".join(c if c.isalnum() else "_" for c in name)


def self_preference_by_judge(judgments, cfg: RunConfig, cluster: bool = True,
                             exclude_refused: bool = False,
                             condition: Condition = Condition.NAIVE) -> pd.DataFrame:
    """Per-judge leniency-adjusted self-preference tau_j.

    Fits  score ~ sum_j 1[j=m & judge=j] + same_family_nonself
                 + C(judge) + C(solver) + C(item),
    so each self coefficient is that judge's own-answer bump net of item
    difficulty, its own leniency (C(judge)), its own answer quality (C(solver)),
    and any same-family favouritism (``same_family_nonself``). The family term
    is what keeps a judge's preference for its sibling out of its
    self-preference estimate. Returns one row per judge with tau, SE, 95% CI, an
    a-priori minimum detectable effect for tau (alpha 0.05, power 0.8), and the
    Holm-adjusted p across the eight judge tests. ``condition`` picks the naive
    or blinded pointwise pass.
    """
    from scipy.stats import norm

    pw = _prep_regression_df(judgments, cfg, exclude_refused=exclude_refused,
                             condition=condition)
    judges = sorted(pw["judge"].unique())
    cols = {}
    for j in judges:
        cols[_san(j)] = ((pw["judge"] == j) & (pw["solver"] == j)).astype(int)
    pw = pw.assign(**cols)
    terms = " + ".join(_san(j) for j in judges)
    formula = (f"parsed_score ~ {terms} + same_family_nonself "
               "+ C(judge) + C(solver) + C(item_id)")
    res = _fit_clustered(formula, pw, cluster)
    mde_factor = norm.ppf(0.975) + norm.ppf(0.8)  # ~2.80 for alpha .05, power .8
    rows = []
    for j in judges:
        k = _san(j)
        tau = float(res.params.get(k, np.nan))
        se = float(res.bse.get(k, np.nan))
        rows.append({"judge": j, "tau": tau, "se": se, "n": int(res.nobs),
                     "ci_lo": tau - 1.96 * se, "ci_hi": tau + 1.96 * se,
                     "mde": mde_factor * se, "p_raw": float(res.pvalues.get(k, np.nan))})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_correction(out["p_raw"].to_numpy())
    return out.sort_values("tau", ascending=False).reset_index(drop=True)


def ordinal_self_preference_by_judge(judgments, cfg: RunConfig,
                                     exclude_refused: bool = False) -> pd.DataFrame:
    """Per-judge self-preference from an ordered-logit model, as a robustness
    check that the linear tau_k are not a floor/ceiling artifact of treating a
    bounded 0--3 ordinal score as continuous.

    Fits a cumulative-link (proportional-odds) model
        P(score <= v) = logit^{-1}(theta_v - eta),
        eta = sum_j b_j 1[j=m=k] + b_fam same_family_nonself
              + judge + solver + item effects,
    so the judge effects shift the latent scale (leniency) while the thresholds
    theta_v are shared. Returns each judge's latent self-preference coefficient
    b_j in log-odds units. What matters is whether their ordering and signs
    match the linear tau_k, not the units.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    pw = _prep_regression_df(judgments, cfg, exclude_refused=exclude_refused)
    judges = sorted(pw["judge"].unique())
    y = pw["parsed_score"].astype(int).to_numpy()
    parts = {}
    for j in judges:
        parts[_san(j)] = ((pw["judge"] == j) & (pw["solver"] == j)).astype(float).to_numpy()
    parts["same_family_nonself"] = pw["same_family_nonself"].astype(float).to_numpy()
    X = pd.DataFrame(parts, index=pw.index)
    for col in ("judge", "solver", "item_id"):
        X = X.join(pd.get_dummies(pw[col], prefix=col, drop_first=True).astype(float))
    res = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False, maxiter=200)
    rows = []
    for j in judges:
        k = _san(j)
        b = float(res.params.get(k, np.nan))
        se = float(res.bse.get(k, np.nan))
        rows.append({"judge": j, "b_logodds": b, "se": se,
                     "p_raw": float(res.pvalues.get(k, np.nan))})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_correction(out["p_raw"].to_numpy())
    return out.sort_values("b_logodds", ascending=False).reset_index(drop=True)


def blinding_paired(judgments, cfg: RunConfig) -> dict:
    """Paired naive-vs-blinded change in per-judge self-preference tau_k.

    Compares each judge's tau_k under the naive and blinded conditions on the
    same judges, with a paired Wilcoxon test on the eight differences and the
    change in between-judge standard deviation. This replaces the mean-absolute
    comparison, whose upward bias in noise made the two conditions look
    different when they are not.
    """
    from scipy.stats import wilcoxon

    nv = self_preference_by_judge(judgments, cfg, condition=Condition.NAIVE)
    bl = self_preference_by_judge(judgments, cfg, condition=Condition.BLINDED)
    m = nv[["judge", "tau"]].merge(bl[["judge", "tau"]], on="judge",
                                   suffixes=("_naive", "_blinded"))
    m["delta"] = m["tau_blinded"] - m["tau_naive"]
    p = np.nan
    d = m["delta"].to_numpy()
    if len(d) >= 3 and np.any(d != 0):
        try:
            p = float(wilcoxon(m["tau_blinded"], m["tau_naive"]).pvalue)
        except ValueError:
            p = np.nan
    return {"per_judge": m.to_dict(orient="records"),
            "mean_delta": float(m["delta"].mean()),
            "sd_naive": float(m["tau_naive"].std(ddof=0)),
            "sd_blinded": float(m["tau_blinded"].std(ddof=0)),
            "paired_p": p}


def family_bias_regression(judgments, cfg: RunConfig, cluster: bool = True,
                           exclude_refused: bool = False) -> object:
    """Same-family bias with solver fixed effects (self-scores excluded):

        score ~ same_family + verbosity + C(judge) + C(solver) + C(item)

    Solver FE absorb answer quality, so ``same_family`` is identified purely
    from the within-answer contrast (in-family vs out-family judge). ``log10_size``
    is deliberately omitted: it is collinear with C(solver) and not identified.
    """
    pw = _prep_regression_df(judgments, cfg, exclude_refused=exclude_refused)
    pw = pw[pw["is_self"] == 0]
    formula = ("parsed_score ~ same_family + verbosity "
               "+ C(judge) + C(solver) + C(item_id)")
    return _fit_clustered(formula, pw, cluster)


def family_bias_by_family(judgments, cfg: RunConfig, cluster: bool = True) -> pd.DataFrame:
    """Family-specific fixed-effects coefficients for multi-model families."""
    pw = _prep_regression_df(judgments, cfg)
    pw = pw[pw["is_self"] == 0].copy()
    families = sorted(
        family for family, n in pw[["solver", "solver_family"]].drop_duplicates()
        ["solver_family"].value_counts().items() if n > 1
    )
    columns = {}
    for family in families:
        columns[f"fam_{family}"] = (
            (pw["judge_family"] == family) & (pw["solver_family"] == family)
        ).astype(int)
    pw = pw.assign(**columns)
    terms = " + ".join(columns)
    formula = (f"parsed_score ~ {terms} + verbosity + C(judge) + C(solver) "
               "+ C(item_id)")
    res = _fit_clustered(formula, pw, cluster)
    return pd.DataFrame([
        {"family": family, "beta": float(res.params[f"fam_{family}"]),
         "se": float(res.bse[f"fam_{family}"]), "n": int(res.nobs)}
        for family in families
    ])


def size_bias_consensus_deviation(judgments, cfg: RunConfig, cluster: bool = True,
                                  exclude_refused: bool = False,
                                  size_measure: str = "total") -> object:
    """Identified size bias via deviation from the leave-one-out panel consensus.

        (score - consensus_{-judge}) ~ log10_size + C(judge) + C(item)

    The dependent variable is how far a judge's score for an answer sits above
    or below the rest of the panel on that same answer, so answer quality is
    differenced out. ``log10_size`` then measures whether a judge systematically
    over-scores larger solvers relative to consensus - a genuine size bias, not
    a size/quality gradient. Self-scores are excluded.
    """
    pw = _prep_regression_df(judgments, cfg, exclude_refused=exclude_refused)
    pw = pw[pw["is_self"] == 0].copy()
    g = pw.groupby(["solver", "item_id"])["parsed_score"]
    grp_sum = g.transform("sum")
    grp_n = g.transform("count")
    pw = pw[grp_n > 1].copy()
    pw["dev"] = pw["parsed_score"] - (grp_sum[grp_n > 1] - pw["parsed_score"]) / (grp_n[grp_n > 1] - 1)
    size_col = "log10_active_size" if size_measure == "active" else "log10_size"
    formula = f"dev ~ {size_col} + C(judge) + C(item_id)"
    return _fit_clustered(formula, pw, cluster)


def _consensus_dev_frame(judgments, cfg: RunConfig,
                         exclude_refused: bool = False) -> pd.DataFrame:
    """Non-self pointwise frame with a leave-one-out consensus-deviation column.

    ``dev`` is a judge's score for an answer minus the mean score the rest of
    the panel gives that same answer, so answer quality is differenced out. Any
    regressor that is constant across judges within an answer (solver size,
    answer length) has a mechanically zero *pooled* slope on this outcome, since
    the deviations sum to zero within each (solver, item) cell. Only the way the
    slope *differs across judges* is identified, which is what the per-judge
    estimators below recover.
    """
    pw = _prep_regression_df(judgments, cfg, exclude_refused=exclude_refused)
    pw = pw[pw["is_self"] == 0].copy()
    g = pw.groupby(["solver", "item_id"])["parsed_score"]
    grp_sum, grp_n = g.transform("sum"), g.transform("count")
    pw = pw[grp_n > 1].copy()
    grp_sum, grp_n = grp_sum[grp_n > 1], grp_n[grp_n > 1]
    pw["dev"] = pw["parsed_score"] - (grp_sum - pw["parsed_score"]) / (grp_n - 1)
    return pw


def _per_judge_slope(pw: pd.DataFrame, regressor: str, cluster: bool) -> pd.DataFrame:
    """Fit  dev ~ sum_j 1[judge=j]*regressor + C(judge) + C(item)  and return
    each judge's slope, cluster-robust SE, CI, plus a joint test that the slopes
    are equal.
    """
    judges = sorted(pw["judge"].unique())
    n = len(judges)
    zcols = {}
    for j in judges:
        zcols[_san(j)] = (pw["judge"] == j).astype(float) * pw[regressor].astype(float)
    pw = pw.assign(**zcols)
    terms = " + ".join(_san(j) for j in judges)
    res = _fit_clustered(f"dev ~ {terms} + C(judge) + C(item_id)", pw, cluster)
    slopes = np.array([float(res.params.get(_san(j), np.nan)) for j in judges])
    grand = np.nanmean(slopes)
    rows = []
    for j, b in zip(judges, slopes, strict=True):
        se = float(res.bse.get(_san(j), np.nan))
        rows.append({"judge": j, "slope": b, "slope_demeaned": b - grand,
                     "se": se, "n": int(res.nobs),
                     "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se,
                     "p_raw": float(res.pvalues.get(_san(j), np.nan))})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_correction(out["p_raw"].to_numpy())
    # joint Wald test that all per-judge slopes are equal
    names = [_san(j) for j in judges]
    joint_p = np.nan
    try:
        R = np.zeros((n - 1, len(res.params)))
        pos = {p: i for i, p in enumerate(res.params.index)}
        for r in range(n - 1):
            R[r, pos[names[r]]] = 1.0
            R[r, pos[names[r + 1]]] = -1.0
        joint_p = float(res.f_test(R).pvalue)
    except Exception:
        pass
    out.attrs["joint_equal_p"] = joint_p
    return out.sort_values("slope", ascending=False).reset_index(drop=True)


def size_bias_by_judge(judgments, cfg: RunConfig, cluster: bool = True,
                       exclude_refused: bool = False,
                       size_measure: str = "total") -> pd.DataFrame:
    """Per-judge size slope on the consensus-deviation outcome (self excluded).

    Replaces the pooled size regression, whose slope is mechanically zero on
    this outcome. Each judge's slope says whether *that judge* scores larger
    solvers above or below the rest of the panel; the joint test asks whether
    the slopes differ across judges. ``slope`` is in rubric points per decade of
    parameters.
    """
    pw = _consensus_dev_frame(judgments, cfg, exclude_refused)
    size_col = "log10_active_size" if size_measure == "active" else "log10_size"
    return _per_judge_slope(pw, size_col, cluster)


def verbosity_bias_by_judge(judgments, cfg: RunConfig, cluster: bool = True,
                            exclude_refused: bool = False) -> pd.DataFrame:
    """Per-judge verbosity slope on the consensus-deviation outcome (self excl.).

    Answer length is standardised so each slope is rubric points per standard
    deviation of length. Like size, the pooled slope is mechanically zero, so
    only the across-judge difference is identified.
    """
    pw = _consensus_dev_frame(judgments, cfg, exclude_refused)
    v = pw["verbosity"].astype(float)
    sd = v.std(ddof=0)
    pw = pw.assign(verbosity_z=(v - v.mean()) / sd if sd > 0 else 0.0)
    return _per_judge_slope(pw, "verbosity_z", cluster)


def size_family_regression(judgments, cfg: RunConfig,
                           anchor_scores: dict | None = None,
                           cluster: bool = True,
                           exclude_refused: bool = False) -> object:
    """OLS with judge & item fixed effects:

        score ~ log10(size) + same_family + verbosity [+ correctness]
                + C(judge) + C(item)

    The ``log10_size`` coefficient (after controlling for quality/verbosity) is
    the size bias; ``same_family`` is the family bias. Item FE absorb
    difficulty and SEs are two-way cluster-robust (item & judge).
    """
    pw = _prep_regression_df(judgments, cfg, anchor_scores, exclude_refused)
    pw = pw[pw["is_self"] == 0]  # exclude self-scores so family bias is not self-preference
    formula = ("parsed_score ~ log10_size + same_family + verbosity "
               "+ C(judge) + C(item_id)")
    if anchor_scores:
        formula += " + correctness"
    return _fit_clustered(formula, pw, cluster)


# per-dataset breakdown

def per_dataset_summary(judgments, cfg: RunConfig, security_only: bool = True,
                        datasets: list[str] | None = None) -> pd.DataFrame:
    """One headline-metrics row per dataset - "on dataset X I measured Y".

    Each judgment carries its source dataset, so the same estimators are run
    within each dataset separately. With ``security_only`` (default) only the
    cyber/security datasets are reported. Robust to small per-dataset slices:
    any metric that can't be computed is left NaN rather than failing the row.
    """
    from judge_bias import aggregate as A
    from judge_bias import mitigate as M

    df = A.to_df(judgments)
    present = datasets or sorted(df["dataset"].dropna().unique())
    rows = []
    for ds in present:
        if security_only and ds not in A.SECURITY_DATASETS:
            continue
        sub = [j for j in judgments if j.dataset == ds]
        sdf = df[df["dataset"] == ds]
        pw = A.pointwise_df(sdf, Condition.NAIVE)
        if pw.empty:
            continue
        spi = A.self_preference_index(pw)
        row = {
            "dataset": ds,
            "domain": A.domain_label(ds),
            "n_items": int(sdf["item_id"].nunique()),
            "n_judges": int(pw["judge"].nunique()),
            "n_pointwise": int(len(pw)),
            "mean_spi": float(spi["spi"].mean()) if not spi.empty else np.nan,
        }
        # canonical fixed-effects self-preference (net of item+judge FE)
        try:
            reg = self_preference_regression(sub, cfg)
            row["fe_self_coef"] = float(reg.params.get("is_self", np.nan))
            row["fe_self_p"] = float(reg.pvalues.get("is_self", np.nan))
        except Exception:
            row["fe_self_coef"] = row["fe_self_p"] = np.nan
        # size / family bias
        try:
            sreg = size_family_regression(sub, cfg)
            row["size_coef"] = float(sreg.params.get("log10_size", np.nan))
            row["family_coef"] = float(sreg.params.get("same_family", np.nan))
        except Exception:
            row["size_coef"] = row["family_coef"] = np.nan
        # ranking volatility + position bias (need pairwise)
        pair = A.pairwise_df(sdf, Condition.NAIVE)
        if not pair.empty:
            try:
                rv = A.ranking_volatility(A.bradley_terry_by_judge(pair))
                row["mean_kendall_tau"] = rv["mean_tau"]
                row["n_distinct_top1"] = rv["n_distinct_top1"]
            except Exception:
                row["mean_kendall_tau"] = row["n_distinct_top1"] = np.nan
            pb = A.position_bias_rate(sdf)
            row["position_flip_rate"] = float(pb["flip_rate"].mean()) if not pb.empty else np.nan
        # refusal rate (security-relevant)
        rb = A.refusal_breakdown(sdf)
        row["refusal_rate"] = float(rb["refusal_rate"].mean()) if not rb.empty else np.nan
        # reliability + blinding mitigation
        try:
            row["krippendorff_alpha"] = krippendorff_alpha(sub)
        except Exception:
            row["krippendorff_alpha"] = np.nan
        try:
            d = M.mitigation_deltas(sub)
            b = d[d["mitigation"] == "blinded"]
            row["blinded_delta_abs_spi"] = float(b["delta_abs_spi"].iloc[0]) if not b.empty else np.nan
        except Exception:
            row["blinded_delta_abs_spi"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


# statistical power / MDE

def power_analysis(judgments, cfg: RunConfig | None = None,
                   alpha: float = 0.05, power: float = 0.8) -> pd.DataFrame:
    """Per-model power / minimum-detectable-effect for the self-preference test.

    Pre-empts the "underpowered" critique: for each model reports the paired-
    sample size, the MDE (smallest self-preference detectable at the target
    power) and the achieved power at the observed effect. Uses the per-item
    paired difference (own − mean-other) as the unit, matching :func:`spi_with_ci`.
    """
    from scipy.stats import norm

    df = to_df(judgments)
    pw = pointwise_df(df, Condition.NAIVE)
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    rows = []
    for m in sorted(set(pw["solver"]) | set(pw["judge"])):
        own = pw[(pw["judge"] == m) & (pw["solver"] == m)]
        oth = pw[(pw["judge"] != m) & (pw["solver"] == m)]
        if own.empty or oth.empty:
            continue
        merged = (own.groupby("item_id")["parsed_score"].mean().to_frame("own")
                  .join(oth.groupby("item_id")["parsed_score"].mean().to_frame("other"),
                        how="inner"))
        d = (merged["own"] - merged["other"]).to_numpy()
        n = len(d)
        sd = float(np.std(d, ddof=1)) if n > 1 else float("nan")
        obs = float(np.mean(d)) if n else float("nan")
        mde = (z_a + z_b) * sd / np.sqrt(n) if n and sd > 0 else float("nan")
        # achieved power at the observed effect (normal approximation)
        ncp = abs(obs) / sd * np.sqrt(n) if n and sd > 0 else 0.0
        achieved = float(1 - norm.cdf(z_a - ncp)) if n and sd > 0 else float("nan")
        rows.append({"model": m, "n_items": n, "sd_diff": sd,
                     "observed_spi": obs, "mde": mde, "achieved_power": achieved,
                     "well_powered": bool(achieved >= power) if n and sd > 0 else False})
    return pd.DataFrame(rows).sort_values("achieved_power").reset_index(drop=True)


# inter-judge reliability

def krippendorff_alpha(judgments, condition: Condition = Condition.NAIVE,
                       level: str = "ordinal") -> float:
    """Krippendorff's α across the judge panel on pointwise scores.

    Reliability units = (item, solver); raters = judges.
    """
    import krippendorff

    df = to_df(judgments)
    pw = pointwise_df(df, condition)
    pw = pw.assign(unit=pw["dataset"] + "|" + pw["item_id"] + "|" + pw["solver"])
    wide = pw.pivot_table(index="judge", columns="unit", values="parsed_score",
                          aggfunc="mean")
    data = wide.to_numpy(dtype=float)
    if data.shape[0] < 2:
        return float("nan")
    # α is undefined when every score is identical (no variance) or there aren't
    # >=2 raters on any unit - common on thin/early runs. Guard instead of crashing.
    finite = data[np.isfinite(data)]
    if finite.size == 0 or np.unique(finite).size < 2:
        return float("nan")
    if (np.isfinite(data).sum(axis=0) >= 2).sum() == 0:
        return float("nan")
    try:
        return float(krippendorff.alpha(reliability_data=data,
                                        level_of_measurement=level))
    except (ValueError, ZeroDivisionError):
        return float("nan")
