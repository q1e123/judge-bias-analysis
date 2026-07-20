"""Per-run log files with a pass/fail report footer.

Wrap a CLI stage's body in ``run_logger("gen")`` and it will:

- tee everything printed to stdout into ``logs/<stage>_<YYYYmmdd-HHMMSS>.log``;
- on exit, append a short run report - elapsed time, ok/error tallies
  scraped from the pool's own ``done: +N ok, M errors`` lines, a sample of the
  per-call ``!! <Error>`` lines, and any exception that aborted the run;
- echo that one-glance verdict (``CLEAN`` / ``HAD ISSUES``) to the console.

So after any run you can open the newest file in ``logs/`` and immediately see
whether it finished cleanly, without scrolling the whole transcript.
"""

from __future__ import annotations

import functools
import io
import re
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

_DONE_RE = re.compile(r"done:\s*\+(\d+)\s*ok,\s*(\d+)\s*errors")
_ERRLINE_RE = re.compile(r"^\s*!!\s")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class _Tee:
    """Fan writes out to several text streams; tolerant of a console whose
    encoding (e.g. Windows cp1252) can't represent every character."""

    def __init__(self, console, *sinks):
        self._console = console
        self._sinks = sinks

    def write(self, s: str):
        try:
            self._console.write(s)
        except UnicodeEncodeError:
            enc = getattr(self._console, "encoding", "ascii") or "ascii"
            self._console.write(s.encode(enc, "replace").decode(enc))
        for sink in self._sinks:
            sink.write(_ANSI_RE.sub("", s))
        return len(s)

    def flush(self):
        for st in (self._console, *self._sinks):
            try:
                st.flush()
            except Exception:  # noqa: BLE001
                pass

    def isatty(self) -> bool:  # keep rich in plain (no-color) mode - clean logs
        return False

    @property
    def encoding(self) -> str:
        return getattr(self._console, "encoding", "utf-8") or "utf-8"


@contextmanager
def run_logger(stage: str, log_dir: Path | str = "logs"):
    """Tee stdout to ``logs/<stage>_<timestamp>.log`` and append a run report."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = log_dir / f"{stage}_{ts}.log"
    fh = path.open("w", encoding="utf-8")
    cap = io.StringIO()
    fh.write(f"=== {stage} @ {datetime.now().isoformat(timespec='seconds')} ===\n")

    console = sys.stdout
    sys.stdout = _Tee(console, fh, cap)
    start = time.perf_counter()
    exc_text: str | None = None
    try:
        yield path
    except BaseException:  # noqa: BLE001 - record, then re-raise
        exc_text = traceback.format_exc()
        raise
    finally:
        sys.stdout = console
        elapsed = time.perf_counter() - start
        text = cap.getvalue()
        ok = sum(int(m.group(1)) for m in _DONE_RE.finditer(text))
        err = sum(int(m.group(2)) for m in _DONE_RE.finditer(text))
        samples = [ln.strip() for ln in text.splitlines() if _ERRLINE_RE.match(ln)][:20]
        clean = err == 0 and exc_text is None
        lines = [
            "",
            "--- run report ---",
            f"stage:    {stage}",
            f"elapsed:  {elapsed:.1f}s",
            f"ok:       {ok}",
            f"errors:   {err}",
            f"verdict:  {'CLEAN' if clean else 'HAD ISSUES'}",
        ]
        if samples:
            lines.append(f"first {len(samples)} error(s):")
            lines += [f"  {s}" for s in samples]
        if exc_text:
            lines += ["", "EXCEPTION (run aborted):", exc_text.rstrip()]
        lines += [f"log file: {path}", ""]
        report = "\n".join(lines) + "\n"
        fh.write(report)
        fh.close()
        # echo only the compact verdict block to the real console
        console.write(report)
        console.flush()


def logged(stage: str):
    """Decorator: run a CLI command inside :func:`run_logger` (signature-preserving
    so Typer still sees the original parameters)."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with run_logger(stage):
                return fn(*args, **kwargs)

        return wrapper

    return deco
