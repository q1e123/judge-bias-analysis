from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from judge_bias import inference
from judge_bias.aggregate import judge_leniency, pointwise_df, to_df
from judge_bias.cache import JudgmentStore
from judge_bias.config import RunConfig, load_registry
from judge_bias.dataset.schema import Condition, JudgeMode
from judge_bias.stats import _prep_regression_df

ANSWER_FE = ("judge", "answer")
LOW_PARSE_JUDGE = "llama-3.1-8b"


def _sha(obj) -> str:
    data = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _taus(ri: dict) -> dict[str, float]:
    return {r["judge"]: r["tau"] for r in ri["per_judge"]}


def _self_summary(ri: dict, reference: dict[str, float] | None = None) -> dict:
    out = {"pooled_tau": ri["pooled_tau"], "pooled_p_ri": ri["pooled_p_ri"],
           "between_judge_sd": ri["between_judge_sd"], "sd_p_ri": ri["sd_p_ri"],
           "per_judge_tau": _taus(ri)}
    if reference is not None:
        judges = sorted(set(reference) & set(out["per_judge_tau"]))
        rho = spearmanr([reference[j] for j in judges],
                        [out["per_judge_tau"][j] for j in judges]).statistic
        out["spearman_vs_original"] = float(rho)
    return out


def _family_summary(ri: dict) -> dict:
    return {"pooled_beta": ri["pooled_beta"], "pooled_p_ri": ri["pooled_p_ri"],
            "per_family": ri["per_family"]}


def missingness(judgments, category: dict[str, str], cfg: RunConfig) -> dict:
    """Parse failure for every judge x arm, a covariate breakdown for the
    high-missingness judge, and a logistic model of its failures."""
    df = to_df(judgments)
    point = df[df["mode"] == JudgeMode.POINTWISE.value]
    pair = df[df["mode"] == JudgeMode.PAIRWISE.value]
    rows = []
    for (cond, j), g in point.groupby(["condition", "judge"]):
        rows.append({"judge": j, "arm": f"pointwise_{cond}", "n": len(g),
                     "n_fail": int(g["parsed_score"].isna().sum())})
    for (cond, j), g in pair.groupby(["condition", "judge"]):
        rows.append({"judge": j, "arm": f"pairwise_{cond}", "n": len(g),
                     "n_fail": int(g["winner"].isna().sum())})
    table = pd.DataFrame(rows)
    table["fail_rate"] = table["n_fail"] / table["n"]

    fam = {m.name: m.family for m in cfg.models}
    g = point[(point["condition"] == Condition.NAIVE.value)
              & (point["judge"] == LOW_PARSE_JUDGE)].copy()
    g["fail"] = g["parsed_score"].isna().astype(int)
    g["category"] = g["item_id"].map(category)
    g["is_self"] = (g["solver"] == LOW_PARSE_JUDGE).astype(int)
    g["same_family_nonself"] = ((g["solver"].map(fam) == fam[LOW_PARSE_JUDGE])
                                & (g["is_self"] == 0)).astype(int)
    g["relation"] = np.select([g["is_self"] == 1, g["same_family_nonself"] == 1],
                              ["self", "same_family"], "other_family")
    g["solver_refused"] = g["solver_refused"].astype(int)
    g["len_quintile"] = pd.qcut(g["answer_len"].rank(method="first"), 5,
                                labels=[f"Q{k}" for k in range(1, 6)])
    g["log_len"] = np.log1p(g["answer_len"].astype(float))

    def rate(col):
        r = g.groupby(col, observed=True)["fail"].agg(["mean", "sum", "count"])
        return {str(k): {"fail_rate": float(v["mean"]), "n_fail": int(v["sum"]),
                         "n": int(v["count"])} for k, v in r.iterrows()}

    breakdown = {c: rate(c) for c in
                 ["solver", "relation", "category", "solver_refused", "len_quintile"]}
    len_by_q = g.groupby("len_quintile", observed=True)["answer_len"].agg(["min", "max"])
    breakdown["len_quintile_bounds"] = {str(k): [int(v["min"]), int(v["max"])]
                                        for k, v in len_by_q.iterrows()}

    logit = smf.logit("fail ~ is_self + same_family_nonself + C(category)"
                      " + solver_refused + log_len", data=g).fit(disp=0)
    logit_out = {k: {"coef": float(logit.params[k]), "se": float(logit.bse[k]),
                     "p": float(logit.pvalues[k])} for k in logit.params.index}
    return {"table": table.to_dict(orient="records"), "llama_breakdown": breakdown,
            "llama_logit": logit_out, "llama_logit_n": int(logit.nobs),
            "_frame": g}


def _impute(judgments, fill) -> list:
    """Copy of ``judgments`` with the low-parse judge's missing naive pointwise
    scores replaced by ``fill(judgment)``."""
    out = []
    for j in judgments:
        if (j.judge == LOW_PARSE_JUDGE and j.mode == JudgeMode.POINTWISE
                and j.condition == Condition.NAIVE and j.parsed_score is None):
            j = j.model_copy(update={"parsed_score": float(fill(j))})
        out.append(j)
    return out


def ipw_pooled_tau(judgments, cfg: RunConfig, frame: pd.DataFrame) -> dict:
    """Pooled tau from Eq. (7) with the low-parse judge's parsed rows weighted
    by the inverse of their fitted parse probability."""
    model = smf.logit("parsed ~ is_self + same_family_nonself + C(category)"
                      " + solver_refused + log_len",
                      data=frame.assign(parsed=1 - frame["fail"])).fit(disp=0)
    p_parse = pd.Series(model.predict(frame), index=frame.index)
    key = dict(zip(zip(frame["solver"], frame["item_id"]), p_parse, strict=True))
    pw = _prep_regression_df(judgments, cfg)
    w = np.ones(len(pw))
    low = (pw["judge"] == LOW_PARSE_JUDGE).to_numpy()
    w[low] = [1.0 / key[(s, i)] for s, i in
              zip(pw.loc[low, "solver"], pw.loc[low, "item_id"], strict=True)]
    fit = smf.wls("parsed_score ~ is_self + same_family_nonself"
                  " + C(judge) + C(solver) + C(item_id)", data=pw, weights=w).fit()
    return {"pooled_tau": float(fit.params["is_self"]),
            "weight_range": [float(w[low].min()), float(w[low].max())]}


def duplicate_audit(inputs_path: Path, responses_path: Path, judgments) -> dict:
    rows = [json.loads(line) for line in inputs_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    by_id = defaultdict(list)
    for r in rows:
        by_id[r["item_id"]].append(r)
    dup = {}
    for iid, recs in by_id.items():
        if len(recs) < 2:
            continue
        dup[iid] = {
            "category": recs[0]["category"],
            "source_prompt_ids": [r["meta"].get("prompt_id") for r in recs],
            "delivered_prompt_sha256": sorted({_sha(r["prompt"]) for r in recs}),
            "full_record_sha256": [_sha(r)[:16] for r in recs],
            "distinct_user_inputs": len({r["meta"].get("user_input") for r in recs}),
        }

    resp = [json.loads(line) for line in responses_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    counts = Counter((r["solver"], r["item_id"]) for r in resp)
    multi = {k for k, v in counts.items() if v > 1}
    last, first = {}, {}
    for r in resp:
        k = (r["solver"], r["item_id"])
        first.setdefault(k, r)
        last[k] = r
    judged_len = {}
    for j in judgments:
        if j.mode == JudgeMode.POINTWISE and j.condition == Condition.NAIVE:
            judged_len[(j.solver, j.item_id)] = j.answer_len
    matches_last = sum(judged_len.get(k) == len(last[k]["text"].split()) for k in multi)
    matches_first = sum(judged_len.get(k) == len(first[k]["text"].split()) for k in multi)
    texts_differ = sum(first[k]["text"] != last[k]["text"] for k in multi)

    jkeys = Counter(j.key() for j in judgments)
    by_cat = Counter(r["category"] for r in rows)
    distinct_by_cat = Counter(recs[0]["category"] for recs in by_id.values())
    return {"records_by_category": dict(by_cat),
            "distinct_ids_by_category": dict(distinct_by_cat),
            "duplicate_ids": dup,
            "n_responses": len(resp), "n_solver_item_pairs": len(counts),
            "n_pairs_with_two_responses": len(multi),
            "solvers_with_two_responses": sorted({s for s, _ in multi}),
            "two_responses_text_differs": texts_differ,
            "judged_matches_last_response": matches_last,
            "judged_matches_first_response": matches_first,
            "max_judgments_per_key": max(jkeys.values()),
            "pi_prompt_contains_user_input": any(
                r["meta"].get("user_input") and r["meta"]["user_input"] in r["prompt"]
                for r in rows if r["category"] == "prompt_injection")}


def practical_scale(judgments, cfg: RunConfig, tau: float) -> dict:
    pw = _prep_regression_df(judgments, cfg)
    nonself = pw[pw["is_self"] == 0]
    within = nonself.groupby("answer")["parsed_score"].std(ddof=1).dropna()
    len_df = judge_leniency(pointwise_df(to_df(judgments), Condition.NAIVE))
    col = "leniency" if "leniency" in len_df.columns else len_df.columns[-1]
    spread = float(len_df[col].max() - len_df[col].min())
    sd_all = float(pw["parsed_score"].std(ddof=1))
    return {"score_sd": sd_all, "tau_over_score_sd": tau / sd_all,
            "mean_within_answer_judge_sd": float(within.mean()),
            "tau_over_within_answer_sd": tau / float(within.mean()),
            "leniency_range": spread, "tau_over_leniency_range": tau / spread,
            "mean_score": float(pw["parsed_score"].mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="workspace_run")
    ap.add_argument("--registry", default="pool8_openrouter.json")
    ap.add_argument("--inputs", default="inputs/pi35x3.jsonl")
    ap.add_argument("--n-perm", type=int, default=10_000)
    args = ap.parse_args()

    cfg = RunConfig(models=load_registry(args.registry), workdir=Path(args.workdir))
    judgments = JudgmentStore(cfg.results_dir / "judgments.jsonl").load()
    items = [json.loads(line) for line in
             (cfg.results_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    category = {it["item_id"]: it["category"] for it in items}
    n = args.n_perm
    out: dict = {"n_perm": n}

    orig_self = inference.self_preference_ri(judgments, cfg, n_perm=n)
    orig_fam = inference.family_bias_ri(judgments, cfg, n_perm=n)
    orig_verb = inference.per_judge_slope_ri(judgments, cfg, "verbosity", n_perm=n)
    ref = _taus(orig_self)
    out["original"] = {"self": _self_summary(orig_self), "family": _family_summary(orig_fam),
                       "verbosity_sd": orig_verb["between_judge_sd"],
                       "verbosity_sd_p": orig_verb["sd_p_ri"]}
    print("original:", out["original"]["self"]["pooled_tau"],
          out["original"]["self"]["pooled_p_ri"], out["original"]["family"]["pooled_p_ri"])

    out["answer_fe"] = {
        "self": _self_summary(inference.self_preference_ri(
            judgments, cfg, n_perm=n, factors=ANSWER_FE), ref),
        "family": _family_summary(inference.family_bias_ri(
            judgments, cfg, n_perm=n, factors=ANSWER_FE)),
    }
    print("answer FE:", out["answer_fe"]["self"]["pooled_tau"],
          out["answer_fe"]["self"]["pooled_p_ri"])

    miss = missingness(judgments, category, cfg)
    frame = miss.pop("_frame")
    out["missingness"] = miss
    pw = _prep_regression_df(judgments, cfg)
    out["effective_n"] = {
        "pointwise_naive_parsed": len(pw),
        "pointwise_naive_parsed_nonself": int((pw["is_self"] == 0).sum()),
        "pointwise_naive_parsed_7judge": int(((pw["judge"] != LOW_PARSE_JUDGE)
                                              & (pw["solver"] != LOW_PARSE_JUDGE)).sum()),
        "pointwise_blinded_parsed": len(_prep_regression_df(
            judgments, cfg, condition=Condition.BLINDED)),
        "pairwise_decided_naive_swapped": int(len(inference._pairwise_both_orders(judgments))),
        "low_parse_judge_self_cells_parsed": int(((pw["judge"] == LOW_PARSE_JUDGE)
                                                  & (pw["is_self"] == 1)).sum()),
    }

    scenarios = {
        "all_0": lambda j: 0,
        "all_3": lambda j: 3,
        "against_finding": lambda j: 0 if j.solver == LOW_PARSE_JUDGE else 3,
        "for_finding": lambda j: 3 if j.solver == LOW_PARSE_JUDGE else 0,
    }
    out["bounds"] = {}
    for name, fill in scenarios.items():
        ri = inference.self_preference_ri(_impute(judgments, fill), cfg, n_perm=n)
        out["bounds"][name] = _self_summary(ri, ref)
        print("bound", name, ri["pooled_tau"], ri["pooled_p_ri"])
    out["ipw"] = ipw_pooled_tau(judgments, cfg, frame)

    js70 = [j for j in judgments if category.get(j.item_id) != "prompt_injection"]
    out["no_pi"] = {
        "n_items": len({j.item_id for j in js70}),
        "self": _self_summary(inference.self_preference_ri(js70, cfg, n_perm=n), ref),
        "self_answer_fe": _self_summary(inference.self_preference_ri(
            js70, cfg, n_perm=n, factors=ANSWER_FE), ref),
        "family": _family_summary(inference.family_bias_ri(js70, cfg, n_perm=n)),
    }
    verb70 = inference.per_judge_slope_ri(js70, cfg, "verbosity", n_perm=n)
    out["no_pi"]["verbosity_sd"] = verb70["between_judge_sd"]
    out["no_pi"]["verbosity_sd_p"] = verb70["sd_p_ri"]
    pair70 = inference.pairwise_self_preference(js70, cfg)
    out["no_pi"]["pairwise_self"] = {"pooled": pair70["pooled"], "p": pair70["p_ri"],
                                     "n_positive": sum(r["self_pref"] > 0
                                                       for r in pair70["per_judge"])}
    print("no PI:", out["no_pi"]["self"]["pooled_tau"], out["no_pi"]["self"]["pooled_p_ri"])

    out["duplicates"] = duplicate_audit(Path(args.inputs),
                                        cfg.results_dir / "responses.jsonl", judgments)
    out["practical"] = practical_scale(judgments, cfg, orig_self["pooled_tau"])

    dest = cfg.results_dir / "revision.json"
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
