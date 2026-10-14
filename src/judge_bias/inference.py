"""Randomization inference and design-honest descriptive statistics.

Eight judge clusters make cluster-robust standard errors approximate, so the
headline effects are re-tested here by permutation instead of by a Wald p-value.
The regression permutations use Frisch-Waugh-Lovell residualization: the fixed
effects are partialled out once, then only the small treatment design is refit
per draw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from judge_bias.aggregate import pairwise_df, to_df
from judge_bias.config import RunConfig
from judge_bias.dataset.schema import Condition, JudgeMode
from judge_bias.stats import _prep_regression_df, holm_correction


def _residualizer(fe: np.ndarray):
    """Return a function that partials the columns of ``fe`` out of any vector."""
    q, _ = np.linalg.qr(fe)

    def resid(v: np.ndarray) -> np.ndarray:
        return v - q @ (q.T @ v)

    return resid


def _fe_design(pw: pd.DataFrame, extra: list[str] | None = None,
               factors: tuple[str, ...] = ("judge", "solver", "item_id")) -> np.ndarray:
    """Intercept + the named factor dummies (first level dropped), plus any
    continuous control columns named in ``extra``. Full column rank so the QR
    projection is clean. ``factors`` omits solver for the size/verbosity models,
    where solver is collinear with the solver-level regressor.
    """
    parts = [np.ones((len(pw), 1))]
    for col in factors:
        parts.append(pd.get_dummies(pw[col], drop_first=True).to_numpy(dtype=float))
    for col in extra or []:
        parts.append(pw[col].to_numpy(dtype=float).reshape(-1, 1))
    return np.column_stack(parts)


def self_preference_ri(judgments, cfg: RunConfig, n_perm: int = 10_000,
                       seed: int = 13,
                       factors: tuple[str, ...] = ("judge", "solver", "item_id")) -> dict:
    """Randomization test for the per-judge self-preference tau_k.

    Within each judge's row the "self" cell is reassigned to a random out-family
    solver, the model is refit, and the null distributions of the pooled tau,
    every tau_k, and their between-judge standard deviation are built. Restricting
    pseudo-self cells to out-family solvers keeps the randomized indicator
    disjoint from the non-self family control, just as it is in the observed
    design. ``factors=("judge", "answer")`` replaces the separate solver and
    item effects with one effect per exact solver answer.
    """
    pw = _prep_regression_df(judgments, cfg)
    judges = sorted(pw["judge"].unique())
    solvers = sorted(pw["solver"].unique())
    family = {m.name: m.family for m in cfg.models}
    choices = {
        j: [m for m in solvers if family.get(m) != family.get(j)]
        for j in judges
    }
    if any(not candidates for candidates in choices.values()):
        raise ValueError("self-preference randomization requires an out-family solver")
    y = pw["parsed_score"].to_numpy(dtype=float)
    # partial out a non-self same-family term too, so tau_k is net of any
    # sibling favouritism (a judge preferring its own family) and not just its
    # own leniency and answer quality
    resid = _residualizer(_fe_design(pw, extra=["same_family_nonself"],
                                     factors=factors))
    ry = resid(y)

    # residualize every candidate (judge j, solver m) self-indicator once
    rc: dict[tuple[str, str], np.ndarray] = {}
    jarr, sarr = pw["judge"].to_numpy(), pw["solver"].to_numpy()
    for j in judges:
        for m in solvers:
            rc[(j, m)] = resid(((jarr == j) & (sarr == m)).astype(float))

    def fit(assign: dict[str, str]) -> np.ndarray:
        X = np.column_stack([rc[(j, assign[j])] for j in judges])
        return np.linalg.lstsq(X, ry, rcond=None)[0]

    def fit_pooled(assign: dict[str, str]) -> float:
        x = np.sum([rc[(j, assign[j])] for j in judges], axis=0)
        return float(x @ ry / (x @ x))

    observed = {j: j for j in judges}
    tau_obs = fit(observed)
    pooled_obs = fit_pooled(observed)
    sd_obs = float(np.std(tau_obs, ddof=0))

    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, len(judges)))
    null_pooled = np.empty(n_perm)
    for p in range(n_perm):
        assign = {j: rng.choice(choices[j]) for j in judges}
        null[p] = fit(assign)
        null_pooled[p] = fit_pooled(assign)
    null_sd = null.std(axis=1, ddof=0)

    p_raw = np.array([(np.sum(np.abs(null[:, k]) >= abs(tau_obs[k])) + 1) / (n_perm + 1)
                      for k in range(len(judges))])
    p_holm = holm_correction(p_raw)
    rows = [{"judge": j, "tau": float(tau_obs[k]), "p_ri": float(p_raw[k]),
             "p_holm": float(p_holm[k])} for k, j in enumerate(judges)]
    out = pd.DataFrame(rows).sort_values("tau", ascending=False).reset_index(drop=True)
    sd_p = (np.sum(null_sd >= sd_obs) + 1) / (n_perm + 1)
    pooled_p = (np.sum(np.abs(null_pooled) >= abs(pooled_obs)) + 1) / (n_perm + 1)
    return {"per_judge": out.to_dict(orient="records"),
            "pooled_tau": pooled_obs, "pooled_p_ri": float(pooled_p),
            "between_judge_sd": sd_obs, "sd_p_ri": float(sd_p),
            "pseudo_self_pool": "out_family", "n_perm": n_perm}


def family_bias_ri(judgments, cfg: RunConfig, n_perm: int = 10_000,
                   seed: int = 17,
                   factors: tuple[str, ...] = ("judge", "solver", "item_id")) -> dict:
    """Permutation test for same-family bias, pooled and per family.

    Family labels are shuffled across the eight models; the same-family
    treatment is rebuilt and the solver-fixed-effects regression refit each
    draw. Reported for the pooled coefficient and for each family with more than
    one member, so no single family's effect leans on the approximate
    cluster-robust p-value. Under ``factors=("judge", "answer")`` the length
    control is dropped because answer length is constant within an answer.
    """
    pw = _prep_regression_df(judgments, cfg)
    pw = pw[pw["is_self"] == 0].copy()
    models = [m.name for m in cfg.models]
    base_fam = {m.name: m.family for m in cfg.models}
    y = pw["parsed_score"].to_numpy(dtype=float)
    extra = [] if "answer" in factors else ["verbosity"]
    resid = _residualizer(_fe_design(pw, extra=extra, factors=factors))
    ry = resid(y)

    jarr, sarr = pw["judge"].to_numpy(), pw["solver"].to_numpy()
    # residualize each ordered model-pair block indicator once (self already gone)
    block: dict[tuple[str, str], np.ndarray] = {}
    for a in models:
        for b in models:
            if a == b:
                continue
            mask = (jarr == a) & (sarr == b)
            if mask.any():
                block[(a, b)] = resid(mask.astype(float))

    def same_family_vec(fam: dict[str, str]) -> np.ndarray:
        cols = [v for (a, b), v in block.items() if fam[a] == fam[b]]
        return np.sum(cols, axis=0) if cols else np.zeros(len(pw))

    def per_family_mat(fam: dict[str, str], fams: list[str]):
        cols = []
        for f in fams:
            members = [v for (a, b), v in block.items()
                       if fam[a] == f and fam[b] == f]
            cols.append(np.sum(members, axis=0) if members else np.zeros(len(pw)))
        return np.column_stack(cols)

    multi = sorted({f for f in base_fam.values()
                    if sum(v == f for v in base_fam.values()) > 1})

    def beta_pooled(fam):
        t = same_family_vec(fam)
        d = float(t @ t)
        return float(t @ ry / d) if d > 0 else np.nan

    def beta_perfamily(fam):
        X = per_family_mat(fam, multi)
        return np.linalg.lstsq(X, ry, rcond=None)[0]

    pooled_obs = beta_pooled(base_fam)
    perfam_obs = beta_perfamily(base_fam)

    rng = np.random.default_rng(seed)
    null_pooled = np.empty(n_perm)
    null_perfam = np.empty((n_perm, len(multi)))
    labels = np.array(models)
    fam_labels = np.array([base_fam[m] for m in models])
    for p in range(n_perm):
        perm = rng.permutation(fam_labels)
        fam = dict(zip(labels, perm, strict=True))
        null_pooled[p] = beta_pooled(fam)
        null_perfam[p] = beta_perfamily(fam)

    pooled_p = (np.sum(np.abs(null_pooled) >= abs(pooled_obs)) + 1) / (n_perm + 1)
    perfam = []
    for k, f in enumerate(multi):
        p_ri = (np.sum(np.abs(null_perfam[:, k]) >= abs(perfam_obs[k])) + 1) / (n_perm + 1)
        perfam.append({"family": f, "beta": float(perfam_obs[k]), "p_ri": float(p_ri)})
    return {"pooled_beta": float(pooled_obs), "pooled_p_ri": float(pooled_p),
            "per_family": perfam, "n_perm": n_perm}


def per_judge_slope_ri(judgments, cfg: RunConfig, kind: str,
                       n_perm: int = 10_000, seed: int = 23) -> dict:
    """Permutation test for per-judge size or verbosity slopes on the
    consensus-deviation outcome.

    The regressor label (solver size, or answer length) is permuted at its
    natural unit and the per-judge slopes are refit by Frisch-Waugh-Lovell
    residualization. This tests whether the slopes differ across judges beyond
    chance, replacing the degenerate cluster-robust Wald test, and gives each
    judge a permutation p-value.
    """
    from judge_bias.stats import _consensus_dev_frame

    pw = _consensus_dev_frame(judgments, cfg).reset_index(drop=True)
    judges = sorted(pw["judge"].unique())
    dev = pw["dev"].to_numpy(dtype=float)
    # no solver FE: solver is collinear with solver-level size (matches
    # size_bias_by_judge / verbosity_bias_by_judge, which use judge + item only)
    resid = _residualizer(_fe_design(pw, factors=("judge", "item_id")))
    rdev = resid(dev)
    jmask = {j: (pw["judge"].to_numpy() == j).astype(float) for j in judges}

    if kind in {"size", "size_active"}:
        size_col = "log10_active_size" if kind == "size_active" else "log10_size"
        base = pw[size_col].to_numpy(dtype=float)
        solvers = sorted(pw["solver"].unique())
        sval = {s: float(pw.loc[pw["solver"] == s, size_col].iloc[0]) for s in solvers}
        svec = np.array([sval[s] for s in solvers])
        srow = pw["solver"].to_numpy()

        def draw(rng):
            perm = dict(zip(solvers, rng.permutation(svec), strict=True))
            return np.array([perm[s] for s in srow])
    elif kind == "verbosity":
        # answer level = (solver, item); groupby and ngroup share the same sorted
        # ordering, so ``ans`` indexes ``alen`` row-for-row
        ans = pw.groupby(["solver", "item_id"]).ngroup().to_numpy()
        adf = (pw.groupby(["solver", "item_id"])["verbosity"].first()
               .reset_index())
        alen = adf["verbosity"].to_numpy(dtype=float)
        raw = pw["verbosity"].to_numpy(dtype=float)
        base = (raw - raw.mean()) / raw.std(ddof=0)
        # permute lengths WITHIN item (across that item's solvers) so the null
        # preserves item difficulty and the solver-count structure and tests only
        # the differential-slope claim (Review R-B8)
        item_groups = [np.array(g.index) for _, g in adf.groupby("item_id")]

        def draw(rng):
            permuted = alen.copy()
            for idx in item_groups:
                permuted[idx] = rng.permutation(alen[idx])
            perm = permuted[ans]
            return (perm - perm.mean()) / perm.std(ddof=0)
    else:
        raise ValueError(kind)

    def slopes(rvec: np.ndarray) -> np.ndarray:
        X = np.column_stack([resid(jmask[j] * rvec) for j in judges])
        return np.linalg.lstsq(X, rdev, rcond=None)[0]

    obs = slopes(base)
    sd_obs = float(np.std(obs, ddof=0))
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, len(judges)))
    for p in range(n_perm):
        null[p] = slopes(draw(rng))
    null_sd = null.std(axis=1, ddof=0)

    rows = []
    for k, j in enumerate(judges):
        p_ri = (np.sum(np.abs(null[:, k]) >= abs(obs[k])) + 1) / (n_perm + 1)
        rows.append({"judge": j, "slope": float(obs[k]), "p_ri": float(p_ri)})
    out = pd.DataFrame(rows).sort_values("slope", ascending=False)
    sd_p = (np.sum(null_sd >= sd_obs) + 1) / (n_perm + 1)
    return {"per_judge": out.to_dict(orient="records"),
            "between_judge_sd": sd_obs, "sd_p_ri": float(sd_p), "n_perm": n_perm}


def _pairwise_both_orders(judgments, conditions=None) -> pd.DataFrame:
    if conditions is None:
        conditions = (Condition.NAIVE, Condition.SWAPPED)
    df = to_df(judgments)
    pair = pd.concat([pairwise_df(df, c) for c in conditions], ignore_index=True)
    return pair[pair["winner"] != "tie"].copy()


def pairwise_self_preference(judgments, cfg: RunConfig, seed: int = 19,
                             conditions=None) -> dict:
    """Per-judge pairwise self-preference, judge-clustered.

    For every decided comparison that includes judge j's own answer, the self
    rate is how often j picks its own answer and the panel rate is how often the
    other judges pick j's answer on the same comparisons. Their difference is
    j's self-preference net of answer quality. The pooled mean is tested by
    sign-flip randomization over the eight judge-level values, so the interval
    reflects the eight clusters rather than the tens of thousands of comparisons.
    ``conditions`` defaults to naive+swapped; pass ``(Condition.BLINDED,)`` for
    the blinded arm.
    """
    pair = _pairwise_both_orders(judgments, conditions)
    judges = sorted(set(pair["judge"]))
    rows = []
    for j in judges:
        inv = pair[(pair["solver_a"] == j) | (pair["solver_b"] == j)]
        win_j = (inv["winner"] == j)
        self_rate = float(win_j[inv["judge"] == j].mean())
        panel_rate = float(win_j[inv["judge"] != j].mean())
        rows.append({"judge": j, "self_rate": self_rate, "panel_rate": panel_rate,
                     "self_pref": self_rate - panel_rate})
    out = pd.DataFrame(rows)
    vals = out["self_pref"].to_numpy(dtype=float)
    pooled = float(vals.mean())
    p_ri = _sign_flip_p(vals, pooled)
    return {"per_judge": out.sort_values("self_pref", ascending=False)
            .to_dict(orient="records"),
            "pooled": pooled, "p_ri": float(p_ri), "n_judges": len(judges)}


def pairwise_family_preference(judgments, cfg: RunConfig) -> dict:
    """Per-judge pairwise in-family preference, judge-clustered (self excluded).

    Restricted to comparisons where exactly one answer shares the judge's family
    and neither answer is the judge's own. The self rate is how often the judge
    picks the same-family answer and the panel rate is how often the other
    judges pick it on the same comparisons.
    """
    pair = _pairwise_both_orders(judgments)
    fam = {m.name: m.family for m in cfg.models}
    judges = sorted(set(pair["judge"]))
    rows = []
    for j in judges:
        jf = fam.get(j)
        sub = pair[(pair["solver_a"] != j) & (pair["solver_b"] != j)].copy()
        fa = sub["solver_a"].map(fam)
        fb = sub["solver_b"].map(fam)
        one_same = (fa == jf) ^ (fb == jf)
        sub = sub[one_same]
        if sub.empty:
            continue
        fa, fb = fa[one_same], fb[one_same]
        picks_same = (((sub["winner"] == sub["solver_a"]) & (fa == jf)) |
                      ((sub["winner"] == sub["solver_b"]) & (fb == jf)))
        self_rate = float(picks_same[sub["judge"] == j].mean()) \
            if (sub["judge"] == j).any() else np.nan
        panel_rate = float(picks_same[sub["judge"] != j].mean()) \
            if (sub["judge"] != j).any() else np.nan
        rows.append({"judge": j, "family": jf, "self_rate": self_rate,
                     "panel_rate": panel_rate,
                     "family_pref": self_rate - panel_rate})
    out = pd.DataFrame(rows).dropna(subset=["family_pref"])
    if out.empty:
        return {"per_judge": [], "pooled": np.nan, "p_ri": np.nan}
    vals = out["family_pref"].to_numpy(dtype=float)
    pooled = float(vals.mean())
    p_ri = _sign_flip_p(vals, pooled)
    return {"per_judge": out.sort_values("family_pref", ascending=False)
            .to_dict(orient="records"),
            "pooled": pooled, "p_ri": float(p_ri), "n_judges": len(out)}


def _sign_flip_null(vals: np.ndarray) -> np.ndarray:
    """Exhaustive (<=12 units) or sampled sign-flip null of the mean."""
    n = len(vals)
    if n <= 12:
        out = np.empty(2 ** n)
        for i in range(2 ** n):
            signs = np.array([1 if (i >> b) & 1 else -1 for b in range(n)])
            out[i] = np.mean(signs * vals)
        return out
    rng = np.random.default_rng(0)
    return np.array([np.mean(rng.choice([-1, 1], n) * vals) for _ in range(4000)])


def _sign_flip_p(vals: np.ndarray, observed: float) -> float:
    """Two-sided sign-flip p, exact when the full sign space is enumerated."""
    null = _sign_flip_null(vals)
    exceed = int(np.sum(np.abs(null) + 1e-12 >= abs(observed)))
    if len(vals) <= 12:
        return exceed / len(null)
    return (exceed + 1) / (len(null) + 1)


def position_direction(judgments) -> dict:
    """Directional first-position (slot A) win rate, over BOTH display orders.

    The flip rate measures inconsistency and has no direction. This reports how
    often the answer shown first is chosen, using the standard both-order
    estimator: for each (item, unordered pair) decided in both orders, slot-A
    preference is (1[AB picks A] + 1[BA picks the answer now in slot A])/2, so a
    judge that always picks the same model scores 0.5 and a purely positional one
    scores 0 or 1. Each (item, pair) is one cluster unit, which avoids treating
    the two orders of a comparison as independent. Per-judge significance is a
    sign-flip permutation over those units, Holm-corrected across judges; the
    pooled test is a judge-clustered sign-flip over the eight judge means.
    """
    df = to_df(judgments)
    naive = pairwise_df(df, Condition.NAIVE)
    swap = pairwise_df(df, Condition.SWAPPED)
    keys = ["dataset", "item_id", "judge", "solver_a", "solver_b"]
    m = naive.merge(swap, on=keys, suffixes=("_ab", "_ba"))
    dec = m[(m["winner_ab"] != "tie") & (m["winner_ba"] != "tie")].copy()
    # slot A holds solver_a under AB and solver_b under BA
    dec["slotA"] = 0.5 * ((dec["winner_ab"] == dec["solver_a"]).astype(float)
                          + (dec["winner_ba"] == dec["solver_b"]).astype(float))
    rows = []
    for j, g in dec.groupby("judge"):
        s = g["slotA"].to_numpy(dtype=float)
        n = len(s)
        p = _sign_flip_p(s - 0.5, float((s - 0.5).mean())) if n else np.nan
        rows.append({"judge": j, "first_win_rate": float(s.mean()) if n else np.nan,
                     "n_units": n, "p": float(p)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_correction(out["p"].to_numpy())
    out = out.sort_values("first_win_rate", ascending=False)
    dev = out["first_win_rate"].to_numpy(dtype=float) - 0.5
    clustered_p = _sign_flip_p(dev, float(dev.mean()))
    return {"per_judge": out.to_dict(orient="records"),
            "pooled_first_win_rate": float(dec["slotA"].mean()) if len(dec) else np.nan,
            "pooled_n_units": len(dec),
            "judge_mean_first_win_rate": float(dev.mean() + 0.5),
            "judge_clustered_p": float(clustered_p),
            "n_judges": len(out)}


def self_vote_rank_impact(judgments, cfg: RunConfig) -> dict:
    """How often a judge's own self-score changes where it ranks its own model.

    For each judge we rank the solvers by that judge's mean pointwise score, then
    re-rank after replacing the judge's scores for its OWN answers with the
    leave-one-out panel consensus on those answers. We report how many judges see
    their own model's rank move, the mean absolute rank move, and how many gain or
    keep the top spot only because of their self-vote. This turns the abstract
    tau=+0.29 into a concrete leaderboard consequence (Review R-B10).
    """
    from judge_bias.aggregate import pointwise_df, to_df

    pw = pointwise_df(to_df(judgments), Condition.NAIVE)
    judges = sorted(set(pw["judge"]))
    changed, moves, top1_changed = 0, [], 0
    for j in judges:
        gj = pw[pw["judge"] == j]
        biased = gj.groupby("solver")["parsed_score"].mean()
        # LOO consensus on j's own answers (other judges scoring solver j)
        own_other = pw[(pw["solver"] == j) & (pw["judge"] != j)]
        debiased = biased.copy()
        if not own_other.empty and j in debiased.index:
            debiased[j] = own_other["parsed_score"].mean()
        rb = biased.rank(ascending=False, method="min")
        rd = debiased.rank(ascending=False, method="min")
        if j in rb.index and j in rd.index:
            dr = int(rd[j] - rb[j])
            moves.append(abs(dr))
            if dr != 0:
                changed += 1
            if (rb[j] == 1) != (rd[j] == 1):
                top1_changed += 1
    return {"n_judges": len(judges), "n_rank_changed": changed,
            "n_top1_changed": top1_changed,
            "mean_abs_rank_move": float(np.mean(moves)) if moves else np.nan,
            "max_abs_rank_move": int(np.max(moves)) if moves else 0}


def per_judge_config_count(judgments, cfg: RunConfig) -> dict:
    """Number of exchangeable pseudo-self configurations available per judge in
    the self-preference randomization (out-family solvers only). Documents the
    coarse per-judge resolution: the smallest attainable per-judge p is about
    1/(count+1) (Review R-A3).
    """
    fam = {m.name: m.family for m in cfg.models}
    solvers = sorted(fam)
    out = {}
    for j in solvers:
        elig = [s for s in solvers if s != j and fam[s] != fam.get(j)]
        out[j] = {"n_out_family": len(elig),
                  "min_attainable_p": 1.0 / (len(elig) + 1)}
    return out


def parse_failure_table(judgments) -> dict:
    """No-parse rates by judge and condition (pointwise) and overall (pairwise).

    Missingness that is concentrated in some judges or conditions changes the
    leave-one-out consensus denominator, so it is reported rather than assumed
    ignorable.
    """
    df = to_df(judgments)
    point = df[df["mode"] == JudgeMode.POINTWISE.value]
    rows = []
    for (cond, j), g in point.groupby(["condition", "judge"]):
        n = len(g)
        fail = int(g["parsed_score"].isna().sum())
        rows.append({"condition": cond, "judge": j, "n": n,
                     "n_fail": fail, "fail_rate": fail / n if n else np.nan})
    per_judge = pd.DataFrame(rows).sort_values(["condition", "fail_rate"],
                                               ascending=[True, False])
    point_overall = {}
    for cond, g in point.groupby("condition"):
        point_overall[cond] = float(g["parsed_score"].isna().mean())
    pair = df[df["mode"] == JudgeMode.PAIRWISE.value]
    pair_overall = {}
    for cond, g in pair.groupby("condition"):
        pair_overall[cond] = float(g["winner"].isna().mean())
    from scipy.stats import pointbiserialr

    naive = point[point["condition"] == Condition.NAIVE.value].copy()
    if "prompt_variant" in naive:
        naive = naive[naive["prompt_variant"] == "default"]
    length_rows = []
    for judge, g in naive.groupby("judge"):
        g = g[g["answer_len"].notna()].copy()
        failed = g["parsed_score"].isna().astype(int)
        rho = p = np.nan
        if len(g) >= 3 and failed.nunique() == 2:
            rho, p = pointbiserialr(failed, g["answer_len"].astype(float))
        length_rows.append({
            "judge": judge, "point_biserial_r": float(rho), "p": float(p),
            "mean_len_failed": float(g.loc[failed == 1, "answer_len"].mean()),
            "mean_len_parsed": float(g.loc[failed == 0, "answer_len"].mean()),
            "n": len(g), "n_fail": int(failed.sum()),
        })
    return {"pointwise_by_judge": per_judge.to_dict(orient="records"),
            "pointwise_overall": point_overall,
            "pairwise_overall": pair_overall,
            "failure_length": length_rows}
