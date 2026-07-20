"""Command-line entry point: ``judge-bias <command>``.

Stages: ``generate`` - ``judge`` - ``anchor`` - ``analyze`` -
``figures``, plus ``run-all`` for an end-to-end (offline-capable) study. Every
stage is resumable through the JSONL caches.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from judge_bias import (
    aggregate,
    figures,
    generate,
    inference,
    judge,
    mitigate,
    provenance,
    stats,
)
from judge_bias.cache import JudgmentStore, ResponseStore
from judge_bias.config import RunConfig, default_registry, load_registry
from judge_bias.dataset.loaders import load_dataset_items
from judge_bias.dataset.schema import TaskItem
from judge_bias.runlog import logged

load_dotenv()
app = typer.Typer(add_completion=False, help="Judge-LLM bias measurement harness.")
console = Console()

def _build_cfg(workdir: Path, pool: str, registry: Path | None,
               pairwise_fraction: float) -> RunConfig:
    models = load_registry(registry) if registry else default_registry(pool)
    return RunConfig(models=models, workdir=workdir,
                     pairwise_fraction=pairwise_fraction)


def _items_path(cfg: RunConfig) -> Path:
    return cfg.results_dir / "items.jsonl"


def _save_items(cfg: RunConfig, items: list[TaskItem]) -> None:
    cfg.ensure_dirs()
    with _items_path(cfg).open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(it.model_dump_json() + "\n")


def _load_items(cfg: RunConfig) -> list[TaskItem]:
    p = _items_path(cfg)
    if not p.exists():
        raise typer.BadParameter("no items.jsonl - run `generate` first")
    return [TaskItem.model_validate_json(ln) for ln in p.read_text("utf-8").splitlines() if ln.strip()]


def _read_items_file(path: Path) -> list[TaskItem]:
    p = Path(path)
    if not p.exists():
        raise typer.BadParameter(f"no such items file: {p}")
    return [TaskItem.model_validate_json(ln)
            for ln in p.read_text("utf-8").splitlines() if ln.strip()]


def _items_sha(items: list[TaskItem]) -> str:
    import hashlib
    h = hashlib.sha256()
    for iid in sorted(f"{it.dataset}|{it.item_id}" for it in items):
        h.update(iid.encode())
    return h.hexdigest()[:16]


def _sample_items(dataset: str, n: int, category: str | None = None,
                  per_category: int | None = None) -> list[TaskItem]:
    if dataset.startswith("synthetic"):
        return load_dataset_items(dataset, n=n)
    kw: dict = {"limit": n}
    if dataset == "cyberseceval":
        if category:
            kw["categories"] = [c.strip() for c in category.split(",") if c.strip()]
        if per_category is not None:
            kw["per_category"] = per_category
            kw.pop("limit", None)  # per-category replaces the total cap
    elif category or per_category is not None:
        console.print("[yellow]--category/--per-category only apply to "
                      "cyberseceval; ignoring[/]")
    return load_dataset_items(dataset, **kw)

@app.command()
def models(pool: str = "oss", registry: Path = None):
    """List the model registry (solver / judge roles, family, size)."""
    cfg = _build_cfg(Path("workspace"), pool, registry, 1.0)
    table = Table(title=f"registry: {pool}")
    for c in ("name", "provider", "family", "size_b", "role"):
        table.add_column(c)
    for m in cfg.models:
        table.add_row(m.name, m.provider, m.family,
                      str(m.size_b), m.role)
    console.print(table)

@app.command()
def check(pool: str = "oss", registry: Path = None):
    """Ping every model with a tiny prompt; report reachable/keys OK."""
    from judge_bias.clients import Message, get_client

    cfg = _build_cfg(Path("workspace"), pool, registry, 1.0)
    table = Table(title=f"provider check: {pool}")
    for c in ("name", "provider", "model_id", "status"):
        table.add_column(c)
    for m in cfg.models:
        try:
            client = get_client(m)
            comp = client.complete([Message("user", "Reply with: ok")],
                                   **{**m.params, "max_tokens": 8})
            ok = "ok" if comp.text.strip() else "empty reply"
            table.add_row(m.name, m.provider, m.model_id, f"[green]{ok}[/]")
        except Exception as e:
            msg = str(e).splitlines()[0][:60]
            table.add_row(m.name, m.provider, m.model_id, f"[red]FAIL[/] {msg}")
    console.print(table)


def _parse_concurrency(spec: str | None) -> dict[str, int] | None:
    """Parse ``--concurrency 'openwebui=1,groq=8'`` into a provider-cap dict.

    Use ``openwebui=1`` for an Ollama-backed OpenWebUI gateway that cold-loads
    one model at a time (keeps the run serial + model-major, no VRAM thrash).
    """
    if not spec:
        return None
    out: dict[str, int] = {}
    for part in spec.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = int(v.strip())
    return out or None


def _select_judges(cfg: RunConfig, spec: str | None):
    """Restrict active judges to a comma-separated name list (None = all).

    Lets you run one backend's judges now and another's later into the same
    store - every judge still scores all solvers, so the union is the full matrix.
    """
    if not spec:
        return None
    want = {s.strip() for s in spec.split(",") if s.strip()}
    chosen = [m for m in cfg.judges if m.name in want]
    missing = want - {m.name for m in chosen}
    if missing:
        console.print(f"[yellow]--judges: unknown name(s) ignored: "
                      f"{sorted(missing)}[/]")
    if not chosen:
        raise typer.BadParameter("--judges matched no models in the registry")
    console.print(f"[dim]active judges: {[m.name for m in chosen]}[/]")
    return chosen


@app.command()
def select(
    out: Path = Path("inputs/frozen.jsonl"),
    dataset: str = "cyberseceval",
    n: int = 40,
    category: str = None,
    per_category: int = None,
):
    """Freeze a deterministic input set to a file, once, for reuse across runs.

    Pick the inputs a single time and write them to ``--out``; then run each
    backend on its own day with ``gen --items-file <out>`` and every run uses
    byte-identical prompts. Prints a SHA so you can confirm two files match.

    Example (balanced 35/category):
      judge-bias select --dataset cyberseceval --per-category 35 --out inputs/pi35x3.jsonl
    """
    import collections

    items = _sample_items(dataset, n, category, per_category)
    if not items:
        console.print("[red]no items selected[/] - check --dataset/--category")
        raise typer.Exit(code=1)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(it.model_dump_json() + "\n")
    cats = dict(collections.Counter(it.category for it in items))
    console.print(f"[green]froze[/] {len(items)} items -> {out}")
    console.print(f"per-category: {cats}")
    console.print(f"inputs sha = {_items_sha(items)}  "
                  f"[dim](same sha = identical inputs)[/]")
    console.print(f"\nnext, on any day / any backend:\n"
                  f"  [cyan]judge-bias gen --items-file {out} --registry <reg.json> ...[/]")


@app.command()
@logged("gen")
def gen(
    dataset: str = "synthetic",
    n: int = 40,
    workdir: Path = Path("workspace"),
    pool: str = "oss",
    registry: Path = None,
    concurrency: str = None,
    category: str = None,
    per_category: int = None,
    items_file: Path = None,
):
    """Generate every solver's answer to every item (resumable).

    For ``--dataset cyberseceval``, ``--category`` scopes to one or more
    CyberSecEval categories (comma-separated): mitre, prompt_injection,
    interpreter_abuse, spear_phishing. ``--n`` then caps items within that scope.

    ``--per-category N`` instead draws a balanced, seeded-random N items from
    *each* included category (stratified across-category design); it overrides
    ``--n``.

    ``--items-file`` loads a frozen input set (from `select`) instead of
    sampling - use it to guarantee byte-identical inputs across days/backends.
    """
    cfg = _build_cfg(workdir, pool, registry, 1.0)
    if items_file is not None:
        items = _read_items_file(items_file)
        console.print(f"[cyan]using frozen inputs[/] {items_file} "
                      f"({len(items)} items, sha {_items_sha(items)})")
    else:
        items = _sample_items(dataset, n, category, per_category)
    _save_items(cfg, items)
    provenance.write_manifest(cfg, items,
                              extra={"dataset": dataset, "category": category,
                                     "per_category": per_category,
                                     "items_file": str(items_file) if items_file else None,
                                     "items_sha": _items_sha(items)})
    resp = generate.generate_all(cfg, items, concurrency=_parse_concurrency(concurrency))
    console.print(f"[green]generated[/] {len(resp)} new responses "
                  f"({len(items)} items x {len(cfg.solvers)} solvers)")


@app.command("judge")
@logged("judge")
def judge_cmd(
    workdir: Path = Path("workspace"),
    pool: str = "oss",
    registry: Path = None,
    pairwise: bool = True,
    pairwise_fraction: float = 1.0,
    concurrency: str = None,
    judges: str = None,
):
    """Run the round-robin judging matrix (pointwise + pairwise).

    ``--judges name1,name2`` restricts which judges run this pass (every judge
    still scores all solvers). Run one backend's judges now and the other's
    later into the same workdir - the union is the full matrix, resumable.
    """
    cfg = _build_cfg(workdir, pool, registry, pairwise_fraction)
    items = _load_items(cfg)
    responses = ResponseStore(cfg.results_dir / "responses.jsonl").load()
    js = judge.judge_all(cfg, items, responses, do_pairwise=pairwise,
                         concurrency=_parse_concurrency(concurrency),
                         judges=_select_judges(cfg, judges))
    console.print(f"[green]produced[/] {len(js)} new judgments")


@app.command()
@logged("anchor")
def anchor(
    n: int = 24,
    source: str = "synthetic",   # synthetic | wmdp | cti-bench
    workdir: Path = Path("workspace"),
    pool: str = "oss",
    registry: Path = None,
    judges: str = None,
):
    """Judge accuracy on a ground-truth anchor (judge-vs-truth agreement).

    source='synthetic' - offline chosen/rejected (synthetic labels);
    source='wmdp' or 'cti-bench' - OBJECTIVE cyber correct-vs-wrong pairs, no
    human labeling, on-domain (use with real pools).

    ``--judges`` restricts which judges run (same split-by-backend semantics as
    the ``judge`` command).
    """
    cfg = _build_cfg(workdir, pool, registry, 1.0)
    ds_name = "synthetic-anchor" if source == "synthetic" else "objective-anchor"
    items = load_dataset_items(ds_name, source=source, n=n)
    js = judge.judge_human_anchor(cfg, items, judges=_select_judges(cfg, judges))
    console.print(f"[green]produced[/] {len(js)} anchor judgments "
                  f"({ds_name}/{source})")


@app.command()
@logged("analyze")
def analyze(
    workdir: Path = Path("workspace"),
    pool: str = "oss",
    registry: Path = None,
):
    """Compute SPI/CIs, bias matrix, rankings, regression, mitigations; write JSON."""
    cfg = _build_cfg(workdir, pool, registry, 1.0)
    judgments = JudgmentStore(cfg.results_dir / "judgments.jsonl").load()
    anchor_path = cfg.results_dir / "anchor.jsonl"
    anchor_js = JudgmentStore(anchor_path).load() if anchor_path.exists() else []

    summary = aggregate.summarise(judgments, cfg, anchor=anchor_js)
    spi_ci = stats.spi_with_ci(judgments)
    summary["spi"] = spi_ci  # richer SPI with CIs + Holm

    # canonical fixed-effects estimators (item+judge FE, cluster-robust SEs)
    sp_reg = stats.self_preference_regression(judgments, cfg)
    reg = stats.size_family_regression(judgments, cfg)
    alpha = stats.krippendorff_alpha(judgments)
    deltas = mitigate.mitigation_deltas(judgments, anchor_naive=anchor_js or None)
    refusal = aggregate.refusal_breakdown(aggregate.to_df(judgments))
    contrast = aggregate.domain_contrast(judgments)
    convergent = aggregate.convergent_validity(judgments)
    power = stats.power_analysis(judgments)
    robust = aggregate.robustness_summary(judgments)

    # refit estimators: per-judge tau, solver-FE family bias, consensus-deviation size
    sp_by_judge = stats.self_preference_by_judge(judgments, cfg)
    fam_fe = stats.family_bias_regression(judgments, cfg)
    fam_by_family = stats.family_bias_by_family(judgments, cfg)
    fam_fe_nr = stats.family_bias_regression(judgments, cfg, exclude_refused=True)
    size_dev = stats.size_bias_consensus_deviation(judgments, cfg)
    size_dev_active = stats.size_bias_consensus_deviation(
        judgments, cfg, size_measure="active"
    )
    size_dev_nr = stats.size_bias_consensus_deviation(judgments, cfg, exclude_refused=True)

    # per-judge size/verbosity slopes (the pooled slope above is mechanically ~0)
    size_by_judge = stats.size_bias_by_judge(judgments, cfg)
    size_by_judge_active = stats.size_bias_by_judge(judgments, cfg, size_measure="active")
    verb_by_judge = stats.verbosity_bias_by_judge(judgments, cfg)

    # randomization inference (8 judge clusters make cluster-robust p-values approximate)
    self_ri = inference.self_preference_ri(judgments, cfg)
    family_ri = inference.family_bias_ri(judgments, cfg)
    size_ri = inference.per_judge_slope_ri(judgments, cfg, "size")
    size_ri_active = inference.per_judge_slope_ri(judgments, cfg, "size_active")
    verb_ri = inference.per_judge_slope_ri(judgments, cfg, "verbosity")
    pairwise_self = inference.pairwise_self_preference(judgments, cfg)
    from judge_bias.dataset.schema import Condition as _CondBl
    pairwise_self_blinded = inference.pairwise_self_preference(
        judgments, cfg, conditions=(_CondBl.BLINDED,))
    pairwise_family = inference.pairwise_family_preference(judgments, cfg)
    position_dir = inference.position_direction(judgments)
    parse_fail = inference.parse_failure_table(judgments)
    blind_paired = stats.blinding_paired(judgments, cfg)

    # refusal reporting (Review R-A5): rates by solver x category, and the
    # self-preference re-estimated on the non-refused subset (the promised sensitivity)
    _pw_ref = aggregate.pointwise_df(aggregate.to_df(judgments), _CondBl.NAIVE)
    try:  # category lives in items.jsonl, not in the judgments
        _cat = {it.item_id: it.category for it in _load_items(cfg)}
    except Exception:
        _cat = {}
    if "solver_refused" in _pw_ref.columns and _cat:
        _pw_ref = _pw_ref.assign(category=_pw_ref["item_id"].map(_cat),
                                 ref=_pw_ref["solver_refused"].astype(bool))
        refusal_by_cat = (_pw_ref.dropna(subset=["category"])
                          .groupby(["solver", "category"])["ref"].mean()
                          .reset_index().rename(columns={"ref": "refusal_rate"})
                          .to_dict(orient="records"))
    else:
        refusal_by_cat = []
    self_by_judge_nonref = stats.self_preference_by_judge(
        judgments, cfg, exclude_refused=True).to_dict(orient="records")

    # Review R-B10 and R-A3 reporting helpers
    rank_impact = inference.self_vote_rank_impact(judgments, cfg)
    judge_config = inference.per_judge_config_count(judgments, cfg)

    # ordinal robustness: does the linear tau ordering survive an ordered logit?
    ordinal = stats.ordinal_self_preference_by_judge(judgments, cfg)

    # M7 sensitivity: drop the one judge with heavy parse failure (llama-3.1-8b)
    drop = "llama-3.1-8b"
    js7 = [j for j in judgments if j.judge != drop and j.solver != drop]
    sens7 = {
        "dropped": drop,
        "self_by_judge": stats.self_preference_by_judge(js7, cfg).to_dict(orient="records"),
        "self_ri": inference.self_preference_ri(js7, cfg),
        "verbosity_ri": inference.per_judge_slope_ri(js7, cfg, "verbosity"),
        "family_ri": inference.family_bias_ri(js7, cfg),
    }
    from judge_bias.dataset.schema import Condition as _CondNv
    _pw_raw = aggregate.pointwise_df(aggregate.to_df(judgments), _CondNv.NAIVE)
    _bm_raw = aggregate.bias_matrix(_pw_raw, row_normalise=False)
    bias_matrix_raw = {j: {s: (None if _bm_raw.loc[j, s] != _bm_raw.loc[j, s]
                               else float(_bm_raw.loc[j, s]))
                           for s in _bm_raw.columns} for j in _bm_raw.index}

    # console report
    console.rule("Self-Preference Index (RQ1)")
    console.print(spi_ci.to_string(index=False))
    console.print(f"\nFE self-preference coef (is_self, net of item+judge FE) = "
                  f"{sp_reg.params.get('is_self', float('nan')):.3f} "
                  f"(p={sp_reg.pvalues.get('is_self', float('nan')):.3g}, "
                  f"cluster-robust)")
    console.rule("Ranking volatility (RQ3)")
    if "ranking_volatility" in summary:
        rv = summary["ranking_volatility"]
        console.print(f"mean Kendall tau across judges: {rv['mean_tau']:.3f}")
        console.print(f"distinct top-1 solvers: {rv['n_distinct_top1']}")
    console.rule("Size/family regression (RQ2)")
    console.print(f"log10_size coef = {reg.params.get('log10_size', float('nan')):.3f} "
                  f"(p={reg.pvalues.get('log10_size', float('nan')):.3g})")
    if "same_family" in reg.params:
        console.print(f"same_family coef = {reg.params['same_family']:.3f} "
                      f"(p={reg.pvalues['same_family']:.3g})")
    console.rule("Refit: per-judge self-preference tau (solver+judge+item FE)")
    console.print(sp_by_judge.to_string(index=False))
    console.rule("Refit: identified family and size bias")
    console.print(f"family (solver FE, self excl.) = {fam_fe.params['same_family']:.3f} "
                  f"(p={fam_fe.pvalues['same_family']:.3g}); "
                  f"non-refused = {fam_fe_nr.params['same_family']:.3f} "
                  f"(p={fam_fe_nr.pvalues['same_family']:.3g})")
    console.print(f"size (consensus-deviation)    = {size_dev.params['log10_size']:.3f} "
                  f"(p={size_dev.pvalues['log10_size']:.3g}); "
                  f"non-refused = {size_dev_nr.params['log10_size']:.3f} "
                  f"(p={size_dev_nr.pvalues['log10_size']:.3g}) [pooled slope is mechanically ~0]")
    console.rule("Randomization inference (honest test for 8 judge clusters)")
    console.print(f"self-preference: between-judge SD = {self_ri['between_judge_sd']:.3f}, "
                  f"SD permutation p = {self_ri['sd_p_ri']:.3f}; pooled tau = "
                  f"{self_ri['pooled_tau']:+.3f} (p={self_ri['pooled_p_ri']:.3f}); "
                  "per-judge RI p:")
    for r in self_ri["per_judge"]:
        console.print(f"  {r['judge']:18} tau={r['tau']:+.2f} p_ri={r['p_ri']:.3f}")
    console.print(f"family: pooled beta={family_ri['pooled_beta']:+.3f} "
                  f"p_ri={family_ri['pooled_p_ri']:.3f}; per family: "
                  + ", ".join(f"{r['family']} {r['beta']:+.2f} (p={r['p_ri']:.3f})"
                              for r in family_ri["per_family"]))
    console.print(f"size differential: SD={size_ri['between_judge_sd']:.3f} "
                  f"p_ri={size_ri['sd_p_ri']:.3f}; "
                  f"verbosity differential: SD={verb_ri['between_judge_sd']:.3f} "
                  f"p_ri={verb_ri['sd_p_ri']:.3f}")
    console.print(f"pairwise self-preference: pooled={pairwise_self['pooled']:+.3f} "
                  f"(judge-clustered p_ri={pairwise_self['p_ri']:.3f})")
    console.print(f"position (first-slot win rate): pooled={position_dir['pooled_first_win_rate']:.3f} "
                  f"(judge-clustered p={position_dir['judge_clustered_p']:.3f})")
    console.print(f"blinding paired: mean delta tau={blind_paired['mean_delta']:+.3f} "
                  f"(p={blind_paired['paired_p']}), SD {blind_paired['sd_naive']:.2f} -> "
                  f"{blind_paired['sd_blinded']:.2f}")
    console.print(f"parse failures (pointwise): {parse_fail['pointwise_overall']}")
    console.rule("Judge-bias diagnostics (verbosity / leniency / scale-use / family)")
    if not summary["verbosity"].empty:
        console.print("verbosity (per-judge Spearman answer-length ~ score):")
        console.print(summary["verbosity"].to_string(index=False))
    if not summary["leniency"].empty:
        console.print("\nleniency/severity (judge mean minus grand mean):")
        console.print(summary["leniency"].to_string(index=False))
    if not summary["scale_usage"].empty:
        console.print("\nscale use (entropy bits / distinct values / std; low = collapsed):")
        console.print(summary["scale_usage"].to_string(index=False))
    fam_ig = summary.get("family_ingroup")
    if fam_ig is not None and not fam_ig.empty:
        console.print("\nfamily in-group bias (same minus other family, self excluded):")
        console.print(fam_ig.to_string(index=False))
    console.rule("Refusal confound (non-refused subset)")
    if not refusal.empty:
        console.print(refusal.to_string(index=False))
    console.rule("Security vs general self-preference (novelty)")
    if contrast.get("per_domain"):
        for dom, v in contrast["per_domain"].items():
            console.print(f"{dom}: mean SPI = {v['mean_spi']:.3f} (n={v['n_models']})")
        if "security_minus_general" in contrast:
            console.print(f"security - general = {contrast['security_minus_general']:.3f} "
                          f"[{contrast['ci_lo']:.3f}, {contrast['ci_hi']:.3f}] (95% CI)")
    else:
        console.print("[dim]single-domain run; add MT-Bench for the contrast[/]")
    console.rule("Prompt robustness (self-preference per judge-prompt variant)")
    if len(robust) > 1:
        console.print(robust.to_string(index=False))
    else:
        console.print("[dim]only the default prompt present; run `judge-bias robust`[/]")
    console.rule("Convergent validity (pointwise vs pairwise)")
    if convergent:
        console.print(f"pooled Spearman = {convergent['pooled_spearman']:.3f}; "
                      f"mean per-judge Spearman = {convergent['mean_spearman']:.3f}")
    console.rule("Statistical power / MDE")
    console.print(power.to_string(index=False))
    console.rule("Inter-judge reliability")
    console.print(f"Krippendorff alpha = {alpha:.3f}")
    console.rule("Mitigations (RQ4)")
    console.print(deltas.to_string(index=False))

    # timing / efficiency (per-model latency from stored latency_s)
    import collections as _collections

    def _lat_table(pairs):
        agg = _collections.defaultdict(list)
        for name, lat in pairs:
            if lat:
                agg[name].append(lat)
        rows = [{"model": k, "n": len(v), "mean_s": round(sum(v) / len(v), 2),
                 "total_s": round(sum(v), 1)} for k, v in sorted(agg.items())]
        return rows

    resp = ResponseStore(cfg.results_dir / "responses.jsonl").load()
    gen_timing = _lat_table((r.solver, r.latency_s) for r in resp)
    judge_timing = _lat_table((j.judge, getattr(j, "latency_s", None)) for j in judgments)
    total_gen_s = round(sum(r.latency_s or 0 for r in resp), 1)
    total_judge_s = round(sum(getattr(j, "latency_s", 0) or 0 for j in judgments), 1)
    def _print_lat(title, rows):
        if not rows:
            return
        console.print(title)
        console.print(f"  {'model':18}{'n':>6}{'mean_s':>9}{'total_s':>10}")
        for r in rows:
            console.print(f"  {r['model']:18}{r['n']:>6}{r['mean_s']:>9}{r['total_s']:>10}")

    console.rule("Timing / efficiency (per-model latency, seconds)")
    console.print(f"total generation compute: {total_gen_s}s across {len(resp)} answers; "
                  f"total judging compute: {total_judge_s}s across {len(judgments)} judgments")
    _print_lat("solver latency (per call):", gen_timing)
    _print_lat("judge latency (per call):", judge_timing)
    console.print("[dim]note: compute-seconds = sum of per-call latencies; wall-clock "
                  "per stage is in the run log (logs/<stage>_*.log).[/]")

    # persist machine-readable summary
    out = cfg.results_dir / "analysis.json"
    # per-solver mean received pointwise score (for the size-bias figure),
    # excluding self-scores so a strong self-preferrer does not inflate its own
    # received mean, with active (not total) parameter counts for the MoE models
    from judge_bias.dataset.schema import Condition as _Cond
    _pw_recv = aggregate.pointwise_df(aggregate.to_df(judgments), _Cond.NAIVE)
    _pw_recv = _pw_recv[_pw_recv["judge"] != _pw_recv["solver"]]
    _size_b = {m.name: m.size_for_regression("total") for m in cfg.models}
    _active_b = {m.name: m.size_for_regression("active") for m in cfg.models}
    _recv = (_pw_recv.groupby("solver")["parsed_score"].mean()
             .reset_index().rename(columns={"parsed_score": "mean_score"}))
    _recv["size_b"] = _recv["solver"].map(_size_b)
    _recv["active_b"] = _recv["solver"].map(_active_b)
    solver_received = _recv.to_dict(orient="records")

    payload = {
        "timing": {
            "total_generation_s": total_gen_s,
            "total_judging_s": total_judge_s,
            "solver_latency": gen_timing,
            "judge_latency": judge_timing,
        },
        "spi": spi_ci.to_dict(orient="records"),
        "self_preference_fe": {
            "is_self_coef": float(sp_reg.params.get("is_self", float("nan"))),
            "is_self_se": float(sp_reg.bse.get("is_self", float("nan"))),
            "is_self_p": float(sp_reg.pvalues.get("is_self", float("nan"))),
            "is_self_p_ri": float(self_ri["pooled_p_ri"]),
            "n": int(sp_reg.nobs),
            "cov_type": "two-way cluster (item, judge), item+judge+solver FE, "
                        "net of same_family_nonself",
        },
        "self_by_judge": sp_by_judge.to_dict(orient="records"),
        "ordinal_self_by_judge": ordinal.to_dict(orient="records"),
        "sensitivity_7judge": sens7,
        "family_bias_solverfe": {
            "coef": float(fam_fe.params["same_family"]),
            "se": float(fam_fe.bse["same_family"]),
            "p": float(fam_fe.pvalues["same_family"]),
            "n": int(fam_fe.nobs),
            "coef_nonrefused": float(fam_fe_nr.params["same_family"]),
            "p_nonrefused": float(fam_fe_nr.pvalues["same_family"]),
        },
        "family_by_family": fam_by_family.to_dict(orient="records"),
        "size_bias_consensus": {
            "coef": float(size_dev.params["log10_size"]),
            "se": float(size_dev.bse["log10_size"]),
            "p": float(size_dev.pvalues["log10_size"]),
            "n": int(size_dev.nobs),
            "active_coef": float(size_dev_active.params["log10_active_size"]),
            "active_se": float(size_dev_active.bse["log10_active_size"]),
            "active_p": float(size_dev_active.pvalues["log10_active_size"]),
            "active_n": int(size_dev_active.nobs),
            "coef_nonrefused": float(size_dev_nr.params["log10_size"]),
            "p_nonrefused": float(size_dev_nr.pvalues["log10_size"]),
            "note": "pooled slope is mechanically zero under balance and negligible here",
        },
        "size_by_judge": size_by_judge.to_dict(orient="records"),
        "size_by_judge_active": size_by_judge_active.to_dict(orient="records"),
        "verbosity_by_judge": verb_by_judge.to_dict(orient="records"),
        "self_preference_ri": self_ri,
        "family_ri": family_ri,
        "size_ri": size_ri,
        "size_ri_active": size_ri_active,
        "verbosity_ri": verb_ri,
        "pairwise_self_preference": pairwise_self,
        "pairwise_self_preference_blinded": pairwise_self_blinded,
        "pairwise_family_preference": pairwise_family,
        "position_direction": position_dir,
        "parse_failure": parse_fail,
        "refusal_by_category": refusal_by_cat,
        "self_by_judge_nonrefused": self_by_judge_nonref,
        "self_vote_rank_impact": rank_impact,
        "per_judge_config_count": judge_config,
        "blinding_paired": blind_paired,
        "bias_matrix_raw": bias_matrix_raw,
        "bias_matrix": summary["bias_matrix"].to_dict(),
        "ranking_volatility": {
            "mean_tau": summary.get("ranking_volatility", {}).get("mean_tau"),
            "n_distinct_top1": summary.get("ranking_volatility", {}).get("n_distinct_top1"),
            "top1_by_judge": summary.get("ranking_volatility", {}).get("top1_by_judge"),
        } if "ranking_volatility" in summary else None,
        "ranking_by_judge": (summary.get("ranking_volatility", {}).get("leaderboards")
                             if "ranking_volatility" in summary else None),
        "solver_received": solver_received,
        "regression": {"params": dict(reg.params), "pvalues": dict(reg.pvalues)},
        "verbosity_bias": summary["verbosity"].to_dict(orient="records"),
        "leniency": summary["leniency"].to_dict(orient="records"),
        "scale_usage": summary["scale_usage"].to_dict(orient="records"),
        "family_ingroup": (summary["family_ingroup"].to_dict(orient="records")
                           if summary.get("family_ingroup") is not None else None),
        "refusal_breakdown": refusal.to_dict(orient="records"),
        "domain_contrast": contrast,
        "prompt_robustness": robust.to_dict(orient="records"),
        "convergent_validity": {k: v for k, v in convergent.items()
                                if k != "per_judge"} if convergent else None,
        "power": power.to_dict(orient="records"),
        "krippendorff_alpha": alpha,
        "krippendorff_method": {
            "level": "ordinal",
            "units": "solver-answer by item",
            "raters": "judges",
            "missing": "pairwise deletion handled by krippendorff 0.6",
        },
        "position_bias": summary["position_bias"].to_dict(orient="records")
            if "position_bias" in summary else None,
        "human_agreement": summary["human_agreement"].to_dict(orient="records")
            if "human_agreement" in summary else None,
        "mitigations": deltas.to_dict(orient="records"),
    }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    console.print(f"\n[green]wrote[/] {out}")


@app.command()
@logged("robust")
def robust(
    workdir: Path = Path("workspace"),
    pool: str = "oss",
    registry: Path = None,
    variants: str = "terse,detailed,scale_1_10",
):
    """Re-judge the pointwise matrix under alternative prompts (robustness sweep)."""
    cfg = _build_cfg(workdir, pool, registry, 1.0)
    items = _load_items(cfg)
    responses = ResponseStore(cfg.results_dir / "responses.jsonl").load()
    js = judge.judge_robustness(cfg, items, responses,
                                variants=[v.strip() for v in variants.split(",")])
    console.print(f"[green]produced[/] {len(js)} robustness judgments")
    rs = aggregate.robustness_summary(
        JudgmentStore(cfg.results_dir / "judgments.jsonl").load())
    console.rule("Self-preference per judge-prompt variant")
    console.print(rs.to_string(index=False))


@app.command()
def figs(
    workdir: Path = Path("workspace"),
    pool: str = "oss",
    registry: Path = None,
):
    """Render the headline figures (bias heatmap, SPI forest, ranking flip)."""
    cfg = _build_cfg(workdir, pool, registry, 1.0)
    judgments = JudgmentStore(cfg.results_dir / "judgments.jsonl").load()
    summary = aggregate.summarise(judgments, cfg)
    summary["spi"] = stats.spi_with_ci(judgments)
    paths = figures.render_all(summary, cfg.figures_dir)
    for p in paths:
        console.print(f"[green]wrote[/] {p}")


@app.command()
def report(
    workdir: Path = Path("workspace"),
    pool: str = "oss",
    registry: Path = None,
    security_only: bool = True,
    datasets: str = "",
):
    """Per-dataset headline metrics - "on dataset X I measured Y" (cyber only by default)."""
    cfg = _build_cfg(workdir, pool, registry, 1.0)
    judgments = JudgmentStore(cfg.results_dir / "judgments.jsonl").load()
    ds_filter = [d.strip() for d in datasets.split(",") if d.strip()] or None
    tbl = stats.per_dataset_summary(judgments, cfg, security_only=security_only,
                                    datasets=ds_filter)
    if tbl.empty:
        console.print("[yellow]no matching datasets in the store[/] - run `gen`/`suite` "
                      "on a security dataset first")
        raise typer.Exit()
    console.rule("Per-dataset results" + (" (security only)" if security_only else ""))
    console.print(tbl.to_string(index=False))
    csv = cfg.results_dir / "per_dataset.csv"
    js = cfg.results_dir / "per_dataset.json"
    tbl.to_csv(csv, index=False)
    js.write_text(tbl.to_json(orient="records", indent=2), encoding="utf-8")
    console.print(f"\n[green]wrote[/] {csv}\n[green]wrote[/] {js}")


@app.command()
@logged("suite")
def suite(
    datasets: str = "synthetic,synthetic-offensive,synthetic-defensive",
    n: int = 40,
    workdir: Path = Path("workspace"),
    pool: str = "oss",
    registry: Path = None,
    pairwise_fraction: float = 1.0,
):
    """Generate + judge across several datasets, then emit the per-dataset report."""
    cfg = _build_cfg(workdir, pool, registry, pairwise_fraction)
    names = [d.strip() for d in datasets.split(",") if d.strip()]
    all_items: list[TaskItem] = []
    for name in names:
        items = load_dataset_items(name, n=n) if name.startswith("synthetic") \
            else load_dataset_items(name)
        console.rule(f"{name}  ({len(items)} items)")
        all_items.extend(items)
        generate.generate_all(cfg, items)
        responses = ResponseStore(cfg.results_dir / "responses.jsonl").load()
        judge.judge_all(cfg, items, responses)
    _save_items(cfg, all_items)
    provenance.write_manifest(cfg, all_items, extra={"datasets": names})
    report(workdir=workdir, pool=pool, registry=registry, security_only=True)


@app.command("run-all")
@logged("run-all")
def run_all(
    dataset: str = "synthetic",
    n: int = 40,
    workdir: Path = Path("workspace"),
    pool: str = "oss",
    registry: Path = None,
    pairwise_fraction: float = 1.0,
    robustness: bool = False,
):
    """End-to-end pipeline: generate - judge - anchor - [robust] - analyze - figures."""
    cfg = _build_cfg(workdir, pool, registry, pairwise_fraction)
    items = load_dataset_items(dataset, n=n) if dataset.startswith("synthetic") \
        else load_dataset_items(dataset, limit=n)
    _save_items(cfg, items)
    provenance.write_manifest(cfg, items, extra={"dataset": dataset})
    generate.generate_all(cfg, items)
    responses = ResponseStore(cfg.results_dir / "responses.jsonl").load()
    judge.judge_all(cfg, items, responses)
    judge.judge_human_anchor(cfg, load_dataset_items("synthetic-anchor", n=24))
    if robustness:
        judge.judge_robustness(cfg, items, responses)
    analyze(workdir=workdir, pool=pool, registry=registry)
    figs(workdir=workdir, pool=pool, registry=registry)


if __name__ == "__main__":
    app()
