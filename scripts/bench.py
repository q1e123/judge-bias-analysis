"""Time gen / judge / analyze on a small sample and extrapolate to a full run.

Every stage's cost scales ~linearly with the number of items (n), so timing a
small n lets you budget a big one. Runs in a throwaway workspace so it always
does real work (never skips cached).

Examples
--------
    # fast path - Groq pool, 10 items, pointwise only
    uv run python scripts/bench.py --pool groq --n 10 --no-pairwise

    # the full OpenRouter pool, 10 items, both modes
    uv run python scripts/bench.py --registry pool8_openrouter.json --n 10 \
        --concurrency openai_compat=8 --dataset cyberseceval

Reports per-stage wall-clock, per-call latency, and a projection table.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv

from judge_bias import aggregate, generate, judge
from judge_bias.cache import JudgmentStore, ResponseStore
from judge_bias.config import RunConfig, default_registry, load_registry
from judge_bias.dataset.loaders import load_dataset_items
from judge_bias.runlog import run_logger


def parse_concurrency(spec: str | None) -> dict[str, int] | None:
    if not spec:
        return None
    return {k.strip(): int(v) for k, v in
            (p.split("=", 1) for p in spec.split(",") if "=" in p)} or None


def fmt(sec: float) -> str:
    if sec < 90:
        return f"{sec:.0f}s"
    if sec < 5400:
        return f"{sec/60:.1f}min"
    return f"{sec/3600:.1f}h"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="oss")
    ap.add_argument("--registry", default=None)
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--pairwise", dest="pairwise", action="store_true", default=True)
    ap.add_argument("--no-pairwise", dest="pairwise", action="store_false")
    ap.add_argument("--pairwise-fraction", type=float, default=1.0)
    ap.add_argument("--concurrency", default=None)
    ap.add_argument("--workdir", default="workspace_bench")
    ap.add_argument("--quiet", action="store_true", help="hide per-call progress")
    args = ap.parse_args()
    verbose = not args.quiet

    load_dotenv()
    models = load_registry(args.registry) if args.registry else default_registry(args.pool)
    workdir = Path(args.workdir)
    shutil.rmtree(workdir, ignore_errors=True)  # fresh - never skip-cached
    cfg = RunConfig(models=models, workdir=workdir,
                    pairwise_fraction=args.pairwise_fraction)
    cfg.ensure_dirs()
    conc = parse_concurrency(args.concurrency)

    print(f"pool: {len(models)} models "
          f"({', '.join(m.name for m in models)})")
    print(f"dataset={args.dataset}  n={args.n}  pairwise={args.pairwise} "
          f"(frac={args.pairwise_fraction})  concurrency={conc or 'defaults'}\n")

    items = (load_dataset_items(args.dataset, n=args.n)
             if args.dataset.startswith("synthetic")
             else load_dataset_items(args.dataset, limit=args.n))
    n = len(items)

    # 1. GENERATE
    print("--- generate ---")
    t = time.perf_counter()
    resp = generate.generate_all(cfg, items, verbose=verbose, concurrency=conc)
    t_gen = time.perf_counter() - t
    n_gen = len(resp)

    # 2. JUDGE
    print("--- judge ---")
    responses = ResponseStore(cfg.results_dir / "responses.jsonl").load()
    t = time.perf_counter()
    js = judge.judge_all(cfg, items, responses, do_pairwise=args.pairwise,
                         verbose=verbose, concurrency=conc)
    t_judge = time.perf_counter() - t
    n_judge = len(js)

    # 3. ANALYZE (headline metric bundle - the CLI adds a few more, all ~constant)
    judgments = JudgmentStore(cfg.results_dir / "judgments.jsonl").load()
    t = time.perf_counter()
    try:
        aggregate.summarise(judgments, cfg)
    except Exception as e:  # noqa: BLE001 - thin data can error; still time it
        print(f"  (analyze note: {type(e).__name__})")
    t_an = time.perf_counter() - t

    total = t_gen + t_judge + t_an
    print("stage      wall-clock   calls   per-call")
    print(f"generate   {fmt(t_gen):>10}   {n_gen:>5}   "
          f"{t_gen/max(n_gen,1):.1f}s")
    print(f"judge      {fmt(t_judge):>10}   {n_judge:>5}   "
          f"{t_judge/max(n_judge,1):.1f}s")
    print(f"analyze    {fmt(t_an):>10}       -   -")
    print(f"TOTAL      {fmt(total):>10}   ({n} items)\n")

    # extrapolation - gen & judge scale linearly in n; analyze ~flat
    print("projected full-run wall-clock (linear in n):")
    print("   n      generate      judge      total")
    for N in (10, 25, 50, 100, 200):
        s = N / max(n, 1)
        g, j = t_gen * s, t_judge * s
        print(f"  {N:>4}   {fmt(g):>10}  {fmt(j):>9}  {fmt(g+j+t_an):>9}")
    print("\nnote: small-n slightly over-estimates per-item on cold-loading "
          "backends (fixed load cost amortizes as n grows).")
    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    with run_logger("bench"):
        main()
