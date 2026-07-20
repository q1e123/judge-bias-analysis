"""Per-backend concurrent execution (efficiency).

Network backends (Groq, OpenWebUI) parallelise well, so we run many requests in
flight up to a per-provider cap. A single-GPU Ollama is compute-bound and
reloads weights into VRAM on every model change, so providers whose cap is <= 1
are run sequentially in submission order - feed them a model-major task list
and each model stays resident instead of thrashing. Serial (GPU) work overlaps
with concurrent (network) work so both resources stay busy.

Every produced result is handed to ``sink`` under a lock, so the callers' JSONL
append + dedup stay thread-safe and resumable.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

# Max in-flight requests per backend. Ollama/local stay ~serial (single GPU).
DEFAULT_CONCURRENCY: dict[str, int] = {
    "groq": 8, "openwebui": 8, "openai_compat": 8,
    "vllm": 2, "local": 2, "ollama": 1,
}


class Task:
    """A unit of work tagged with the provider whose rate limit it draws on
    (and an optional model name, shown in verbose progress)."""

    __slots__ = ("provider", "fn", "name")

    def __init__(self, provider: str, fn: Callable, name: str | None = None):
        self.provider = provider
        self.fn = fn
        self.name = name


def _cap(conc: dict[str, int], provider: str) -> int:
    return conc.get(provider, 4)


def _fmt(sec: float) -> str:
    if sec < 90:
        return f"{sec:.0f}s"
    if sec < 5400:
        return f"{sec / 60:.1f}min"
    return f"{sec / 3600:.1f}h"


def _safe(fn: Callable, verbose: bool):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - one failed call must not kill the batch
        if verbose:
            print(f"  !! {type(e).__name__}: {str(e).splitlines()[0][:100]}")
        return None


def run_pool(tasks: list[Task], sink: Callable, *,
             concurrency: dict[str, int] | None = None,
             verbose: bool = True, label: str = "tasks") -> dict:
    """Execute ``t.fn()`` for each task; call ``sink(result)`` for each non-None.

    Providers with cap <= 1 run sequentially in submission order (model-major);
    the rest run concurrently under a per-provider semaphore. When ``verbose``,
    prints a live progress line per completion (throttled for large batches):
    ``[judge] 42/140  3.1min elapsed  ~7.3min left  (ok 42, err 0)  last: qwen``.
    Returns counters.
    """
    conc = {**DEFAULT_CONCURRENCY, **(concurrency or {})}
    lock = threading.Lock()
    total = len(tasks)
    counters = {"ok": 0, "err": 0, "total": total}
    start = time.perf_counter()

    def _progress(last: str | None) -> None:
        done = counters["ok"] + counters["err"]
        # every completion for small batches; sampled (~40 ticks) for large ones
        if done != total and total > 80 and done % max(1, total // 40):
            return
        el = time.perf_counter() - start
        rate = done / el if el > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        print(f"[{label}] {done}/{total}  {_fmt(el)} elapsed  ~{_fmt(eta)} left  "
              f"(ok {counters['ok']}, err {counters['err']})"
              + (f"  last: {last}" if last else ""), flush=True)

    def emit(result, task: Task | None = None) -> None:
        with lock:
            if result is None:
                counters["err"] += 1
            else:
                sink(result)
                counters["ok"] += 1
            if verbose:
                _progress(getattr(task, "name", None)
                          or (task.provider if task else None))

    serial = [t for t in tasks if _cap(conc, t.provider) <= 1]
    concurrent = [t for t in tasks if _cap(conc, t.provider) > 1]
    sems = {t.provider: threading.Semaphore(_cap(conc, t.provider)) for t in concurrent}

    def run_concurrent(t: Task):
        with sems[t.provider]:
            return t, _safe(t.fn, verbose)

    def run_serial() -> None:  # in submission order - model-major, no VRAM thrash
        for t in serial:
            emit(_safe(t.fn, verbose), t)

    if not tasks:
        if verbose:
            print(f"[pool] {label}: nothing to do")
        return counters

    if verbose:
        sp = sorted({t.provider for t in serial})
        cp = sorted({t.provider for t in concurrent})
        print(f"[{label}] {total} calls -- serial {len(serial)} {sp or '-'}, "
              f"concurrent {len(concurrent)} {cp or '-'}", flush=True)

    max_workers = min(32, max(1, sum(_cap(conc, p) for p in sems))) if sems else 1
    with ThreadPoolExecutor(max_workers=max_workers + 1) as ex:
        serial_future = ex.submit(run_serial) if serial else None
        futures = [ex.submit(run_concurrent, t) for t in concurrent]
        for fut in as_completed(futures):
            t, res = fut.result()
            emit(res, t)
        if serial_future is not None:
            serial_future.result()

    if verbose:
        print(f"[{label}] done: +{counters['ok']} ok, {counters['err']} errors "
              f"of {total} in {_fmt(time.perf_counter() - start)}", flush=True)
    return counters
