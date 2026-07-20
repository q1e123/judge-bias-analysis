"""Round-robin judging: pointwise + pairwise, with position-swap and blinding.

Every judge scores every solver's answer on every item. Conditions produced:

- pointwise ``NAIVE``   : identity visible
- pointwise ``BLINDED`` : solver identity / style tells stripped
- pairwise  ``NAIVE``   : pair shown in order AB, identity visible
- pairwise  ``SWAPPED`` : same pair shown in order BA (for position-bias rate)
- pairwise  ``BLINDED`` : order AB, identity stripped

All judgments are appended to one JSONL store, keyed for resume.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable

from judge_bias.cache import JudgmentStore
from judge_bias.clients import Message, get_client
from judge_bias.config import ModelSpec, RunConfig
from judge_bias.dataset.schema import (
    Condition,
    JudgeMode,
    Judgment,
    Response,
    TaskItem,
)
from judge_bias.pool import Task, run_pool
from judge_bias.prompts import (
    PAIRWISE_SYSTEM,
    POINTWISE_SYSTEM,
    PromptVariant,
    get_pointwise_variant,
    render_pairwise,
    render_pointwise,
)

ID_FOOTER_RE = re.compile(r"<<ID:[^>]*>>")
SCORE_RE = re.compile(r"score\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
WINNER_RE = re.compile(r"winner\s*[:=]\s*(a|b|tie)", re.IGNORECASE)


# blinding

def blind_answer(text: str, model_names: Iterable[str]) -> str:
    """Strip identity tells: any embedded ID footer and any literal model name."""
    text = ID_FOOTER_RE.sub("", text)
    for name in model_names:
        if name:
            text = re.sub(re.escape(name), "the model", text, flags=re.IGNORECASE)
    return text.strip()


# parsing

def parse_score(text: str, lo: int, hi: int) -> tuple[float | None, bool]:
    """Return (clamped_score, refused). refused=True if unparseable."""
    m = SCORE_RE.search(text)
    if not m:
        return None, True
    val = float(m.group(1))
    return max(lo, min(hi, val)), False


def parse_winner(text: str) -> str | None:
    m = WINNER_RE.search(text)
    if not m:
        return None
    return m.group(1).upper() if m.group(1).lower() != "tie" else "tie"


# pointwise

def judge_pointwise_one(
    judge: ModelSpec, item: TaskItem, resp: Response,
    condition: Condition, model_names: list[str],
    variant: PromptVariant | None = None,
) -> Judgment:
    answer = resp.text
    if condition == Condition.BLINDED:
        answer = blind_answer(answer, model_names)
    if variant is None:
        rubric = item.rubric
        lo = rubric.scale_min if rubric else 0
        hi = rubric.scale_max if rubric else 3
        system = POINTWISE_SYSTEM
        prompt = render_pointwise(item.prompt, answer, rubric)
        variant_name = "default"
    else:
        lo, hi = variant.scale_min, variant.scale_max
        system = variant.system
        prompt = variant.render(item.prompt, answer, item.rubric)
        variant_name = variant.name
    client = get_client(judge)
    comp = client.complete(
        [Message("system", system), Message("user", prompt)],
        _ctx={"kind": "judge", "mode": "pointwise", "condition": condition.value},
        **judge.params,
    )
    score, refused = parse_score(comp.text, lo, hi)
    return Judgment(
        dataset=item.dataset, item_id=item.item_id, judge=judge.name,
        mode=JudgeMode.POINTWISE, condition=condition, solver=resp.solver,
        parsed_score=score, raw_output=comp.text, refused=refused,
        solver_refused=resp.refused, prompt_variant=variant_name,
        answer_len=len(answer.split()), latency_s=comp.latency_s,
        tokens_in=comp.tokens_in, tokens_out=comp.tokens_out,
    )


# pairwise

def _winner_to_solver(disp: str | None, order: str, s_a: str, s_b: str) -> str | None:
    if disp is None:
        return None
    if disp == "tie":
        return "tie"
    # order "AB": display A=s_a, B=s_b ; order "BA": display A=s_b, B=s_a
    if order == "AB":
        return s_a if disp == "A" else s_b
    return s_b if disp == "A" else s_a


def judge_pairwise_one(
    judge: ModelSpec, item: TaskItem, ra: Response, rb: Response,
    order: str, condition: Condition, model_names: list[str],
) -> Judgment:
    a_text, b_text = ra.text, rb.text
    if condition == Condition.BLINDED:
        a_text = blind_answer(a_text, model_names)
        b_text = blind_answer(b_text, model_names)
    # arrange display order
    if order == "AB":
        disp_a, disp_b = a_text, b_text
    else:
        disp_a, disp_b = b_text, a_text
    prompt = render_pairwise(item.prompt, disp_a, disp_b)
    client = get_client(judge)
    comp = client.complete(
        [Message("system", PAIRWISE_SYSTEM), Message("user", prompt)],
        _ctx={"kind": "judge", "mode": "pairwise", "condition": condition.value},
        **judge.params,
    )
    disp = parse_winner(comp.text)
    winner = _winner_to_solver(disp, order, ra.solver, rb.solver)
    return Judgment(
        dataset=item.dataset, item_id=item.item_id, judge=judge.name,
        mode=JudgeMode.PAIRWISE, condition=condition,
        solver_a=ra.solver, solver_b=rb.solver, order=order, winner=winner,
        raw_output=comp.text, refused=disp is None,
        latency_s=comp.latency_s, tokens_in=comp.tokens_in,
        tokens_out=comp.tokens_out,
    )


# driver

def judge_human_anchor(
    cfg: RunConfig, items: Iterable[TaskItem],
    store: JudgmentStore | None = None,
    condition: Condition = Condition.NAIVE, verbose: bool = True,
    judges: Iterable[ModelSpec] | None = None,
) -> list[Judgment]:
    """Judge each model on human chosen/rejected pairs - judge-vs-human accuracy.

    Produces pairwise judgments with solver_a='__chosen__',
    solver_b='__rejected__'; a correct judgment picks '__chosen__'. The
    presentation order is randomised per item so position bias does not inflate
    accuracy. Feeds :func:`judge_bias.aggregate.human_agreement`.
    """
    import hashlib

    anchored = [it for it in items
                if it.reference and it.reference.chosen and it.reference.rejected]
    model_names = [m.name for m in cfg.models]
    if store is None:
        cfg.ensure_dirs()
        store = JudgmentStore(cfg.results_dir / "anchor.jsonl")
    done = store.existing_keys()
    out: list[Judgment] = []
    active_judges = list(judges) if judges is not None else cfg.judges
    for judge in active_judges:
        n_new = 0
        for it in anchored:
            h = int(hashlib.sha256((judge.name + it.item_id).encode()).hexdigest()[:8], 16)
            order = "AB" if h % 2 == 0 else "BA"
            chosen, rejected = it.reference.chosen, it.reference.rejected
            if condition == Condition.BLINDED:
                chosen = blind_answer(chosen, model_names)
                rejected = blind_answer(rejected, model_names)
            disp_a, disp_b = (chosen, rejected) if order == "AB" else (rejected, chosen)
            probe = Judgment(dataset=it.dataset, item_id=it.item_id, judge=judge.name,
                             mode=JudgeMode.PAIRWISE, condition=condition,
                             solver_a="__chosen__", solver_b="__rejected__", order=order)
            if probe.key() in done:
                continue
            prompt = render_pairwise(it.prompt, disp_a, disp_b)
            client = get_client(judge)
            comp = client.complete(
                [Message("system", PAIRWISE_SYSTEM), Message("user", prompt)],
                _ctx={"kind": "judge", "mode": "pairwise", "condition": condition.value},
                **judge.params,
            )
            disp = parse_winner(comp.text)
            winner = _winner_to_solver(disp, order, "__chosen__", "__rejected__")
            j = Judgment(dataset=it.dataset, item_id=it.item_id, judge=judge.name,
                         mode=JudgeMode.PAIRWISE, condition=condition,
                         solver_a="__chosen__", solver_b="__rejected__", order=order,
                         winner=winner, raw_output=comp.text, refused=disp is None,
                         latency_s=comp.latency_s, tokens_in=comp.tokens_in,
                         tokens_out=comp.tokens_out)
            store.append(j)
            done.add(j.key())
            out.append(j)
            n_new += 1
        if verbose:
            print(f"[judge:anchor] {judge.name}: +{n_new}")
    return out


def _select_pairs(solvers: list[str], item_id: str, fraction: float) -> list[tuple[str, str]]:
    pairs = list(itertools.combinations(sorted(solvers), 2))
    if fraction >= 1.0:
        return pairs
    import hashlib

    def keep(p):
        h = hashlib.sha256(f"{item_id}|{p[0]}|{p[1]}".encode()).hexdigest()
        return (int(h[:8], 16) / float(16**8)) < fraction

    return [p for p in pairs if keep(p)]


def judge_all(
    cfg: RunConfig,
    items: Iterable[TaskItem],
    responses: Iterable[Response],
    store: JudgmentStore | None = None,
    do_pairwise: bool = True,
    verbose: bool = True,
    concurrency: dict[str, int] | None = None,
    judges: Iterable[ModelSpec] | None = None,
) -> list[Judgment]:
    """Run the round-robin. Resumable: skips judgments already cached.

    Uses per-backend concurrency: the task list is judge-major so a
    single-GPU Ollama judge stays resident; network judges run in parallel.

    ``judges`` restricts *which* judges are active this run (defaults to all
    ``cfg.judges``); every judge still scores all solvers' answers, so
    running one backend's judges now and another's later builds the same full
    matrix into one store - resumable and dedup-safe.
    """
    items = list(items)
    items_by_id = {(it.dataset, it.item_id): it for it in items}
    resp_by_key = {(r.dataset, r.item_id, r.solver): r for r in responses}
    model_names = [m.name for m in cfg.models]
    active_judges = list(judges) if judges is not None else cfg.judges

    cfg.ensure_dirs()
    store = store or JudgmentStore(cfg.results_dir / "judgments.jsonl")
    done = store.existing_keys()
    out: list[Judgment] = []

    pointwise_conditions = [Condition.NAIVE]
    if cfg.run_blinded:
        pointwise_conditions.append(Condition.BLINDED)

    tasks: list[Task] = []

    # pointwise round-robin (judge-major: keeps an Ollama judge resident)
    for judge in active_judges:
        for (ds, iid), item in items_by_id.items():
            for solver in cfg.solvers:
                r = resp_by_key.get((ds, iid, solver.name))
                if r is None:
                    continue
                for cond in pointwise_conditions:
                    jd = Judgment(dataset=ds, item_id=iid, judge=judge.name,
                                  mode=JudgeMode.POINTWISE, condition=cond,
                                  solver=solver.name)
                    if jd.key() in done:
                        continue
                    tasks.append(Task(
                        judge.provider,
                        (lambda j=judge, it=item, rr=r, c=cond:
                         judge_pointwise_one(j, it, rr, c, model_names)),
                        name=judge.name,
                    ))

    # pairwise round-robin
    if do_pairwise:
        pairwise_plan = [("AB", Condition.NAIVE)]
        if cfg.run_swapped:
            pairwise_plan.append(("BA", Condition.SWAPPED))
        if cfg.run_blinded:
            pairwise_plan.append(("AB", Condition.BLINDED))
        solver_names = [s.name for s in cfg.solvers]
        for judge in active_judges:
            for (ds, iid), item in items_by_id.items():
                for s_a, s_b in _select_pairs(solver_names, iid, cfg.pairwise_fraction):
                    ra = resp_by_key.get((ds, iid, s_a))
                    rb = resp_by_key.get((ds, iid, s_b))
                    if ra is None or rb is None:
                        continue
                    for order, cond in pairwise_plan:
                        probe = Judgment(dataset=ds, item_id=iid, judge=judge.name,
                                         mode=JudgeMode.PAIRWISE, condition=cond,
                                         solver_a=s_a, solver_b=s_b, order=order)
                        if probe.key() in done:
                            continue
                        tasks.append(Task(
                            judge.provider,
                            (lambda j=judge, it=item, a=ra, b=rb, o=order, c=cond:
                             judge_pairwise_one(j, it, a, b, o, c, model_names)),
                            name=judge.name,
                        ))

    def sink(j: Judgment) -> None:
        store.append(j)
        out.append(j)

    run_pool(tasks, sink, concurrency=concurrency, verbose=verbose, label="judge")
    return out


def judge_robustness(
    cfg: RunConfig,
    items: Iterable[TaskItem],
    responses: Iterable[Response],
    variants: list[str] | None = None,
    store: JudgmentStore | None = None,
    verbose: bool = True,
) -> list[Judgment]:
    """Re-run the pointwise NAIVE matrix under alternative judge prompts.

    Demonstrates the bias result is not an artefact of one prompt wording or
    score scale (the key robustness check for any LLM-judge paper). Writes
    judgments tagged with ``prompt_variant`` into the same store; resumable.
    """
    variants = variants or [v for v in ("terse", "detailed", "scale_1_10")]
    items = list(items)
    items_by_id = {(it.dataset, it.item_id): it for it in items}
    resp_by_key = {(r.dataset, r.item_id, r.solver): r for r in responses}
    model_names = [m.name for m in cfg.models]

    cfg.ensure_dirs()
    store = store or JudgmentStore(cfg.results_dir / "judgments.jsonl")
    done = store.existing_keys()
    out: list[Judgment] = []

    for vname in variants:
        variant = get_pointwise_variant(vname)
        for judge in cfg.judges:
            n_new = 0
            for (ds, iid), item in items_by_id.items():
                for solver in cfg.solvers:
                    r = resp_by_key.get((ds, iid, solver.name))
                    if r is None:
                        continue
                    probe = Judgment(dataset=ds, item_id=iid, judge=judge.name,
                                     mode=JudgeMode.POINTWISE, condition=Condition.NAIVE,
                                     solver=solver.name, prompt_variant=vname)
                    if probe.key() in done:
                        continue
                    j = judge_pointwise_one(judge, item, r, Condition.NAIVE,
                                            model_names, variant=variant)
                    store.append(j)
                    done.add(j.key())
                    out.append(j)
                    n_new += 1
            if verbose:
                print(f"[judge:robust:{vname}] {judge.name}: +{n_new}")
    return out
