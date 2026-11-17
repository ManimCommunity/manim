"""Out-of-process rendering for the ``manim`` Sphinx directive.

Rendering an example inside the Sphinx process has two failure modes that are
hard to diagnose on CI builders: memory that native libraries (video encoder,
Cairo, Pango) keep around accumulates over hundreds of examples, and a render
that hangs takes the whole documentation build with it, without any output.

This module renders examples in a separate worker process instead:

- The worker is reused across examples (importing manim is expensive), but it
  is replaced after a fixed number of renders or once its resident memory
  exceeds a threshold. All memory a worker accumulated is freed at that point.
- Each render has a wall-clock timeout. When it is exceeded, the stack of every
  thread in the worker is dumped to stderr via :mod:`faulthandler` and the
  worker is killed, so the build fails with a clear error instead of hanging.

The worker is started with the ``spawn`` method so it does not inherit any
Sphinx state, threads, or locks from the parent process.

Configuration (environment variables, mostly useful for CI):

``MANIM_DOCS_RENDER_TIMEOUT``
    Seconds a single example may take before the worker is killed
    (default: 300).
``MANIM_DOCS_WORKER_MAX_RENDERS``
    Number of examples after which a worker is replaced (default: 50).
``MANIM_DOCS_WORKER_MAX_RSS_MB``
    Resident memory in MB above which a worker is replaced after finishing its
    current example (default: 1024).
``MANIM_DOCS_RENDER_IN_PROCESS``
    If set to ``1``, render in the Sphinx process as before (no isolation).
"""

from __future__ import annotations

import contextlib
import faulthandler
import gc
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["RenderError", "RenderResult", "render_example", "shutdown"]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


RENDER_TIMEOUT = _env_float("MANIM_DOCS_RENDER_TIMEOUT", 300)
MAX_RENDERS_PER_WORKER = int(_env_float("MANIM_DOCS_WORKER_MAX_RENDERS", 50))
MAX_WORKER_RSS_MB = _env_float("MANIM_DOCS_WORKER_MAX_RSS_MB", 1024)
RENDER_IN_PROCESS = os.environ.get("MANIM_DOCS_RENDER_IN_PROCESS") == "1"


class RenderError(RuntimeError):
    """Raised when an example fails to render, times out, or kills its worker."""


@dataclass(frozen=True)
class RenderResult:
    """Outcome of rendering one example."""

    file_path: Path
    run_time: float


@dataclass(frozen=True)
class _Job:
    clsname: str
    code: str
    global_config: dict[str, Any]
    example_config: dict[str, Any]


def _rss_mb() -> float:
    """Resident memory of the current process in MB (Linux), else 0."""
    try:
        with open("/proc/self/statm") as statm:
            return int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError, AttributeError):
        return 0.0


def _release_memory() -> None:
    """Return freed heap memory to the operating system where supported."""
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


def _render(job: _Job) -> RenderResult:
    """Render one example in the current process."""
    from manim import config, tempconfig

    for key, value in job.global_config.items():
        setattr(config, key, value)

    namespace: dict[str, Any] = {}
    code = "\n".join(
        [
            "from manim import *",
            job.code,
            f"_manim_rendered_scene = {job.clsname}()",
            "_manim_rendered_scene.render()",
        ]
    )
    with tempconfig(job.example_config):
        start = time.perf_counter()
        exec(compile(code, f"<manim example {job.clsname}>", "exec"), namespace)
        run_time = time.perf_counter() - start
        scene = namespace.pop("_manim_rendered_scene")
        file_path = Path(scene.renderer.file_writer.final_file_path)
    del scene, namespace
    _release_memory()
    return RenderResult(file_path=file_path, run_time=run_time)


def _worker_main(conn: Any) -> None:
    """Entry point of the worker process: render jobs until told to stop."""
    # Tracebacks of all threads when the parent sends SIGABRT before killing
    # us, or on a fatal native error.
    faulthandler.enable(file=sys.stderr, all_threads=True)
    # Thread stacks on demand (``kill -USR1 <pid>``), without stopping.
    if hasattr(faulthandler, "register") and hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    while True:
        try:
            job = conn.recv()
        except EOFError:
            return
        if job is None:
            return
        try:
            result = _render(job)
            conn.send(("ok", result, _rss_mb()))
        except BaseException as exc:
            conn.send(("error", "".join(traceback.format_exception(exc)), _rss_mb()))


class _Worker:
    def __init__(self) -> None:
        ctx = mp.get_context("spawn")
        self.conn, child_conn = ctx.Pipe()
        self.process = ctx.Process(
            target=_worker_main,
            args=(child_conn,),
            name="manim-docs-render-worker",
            daemon=True,
        )
        self.process.start()
        child_conn.close()
        self.renders = 0
        self.rss_mb = 0.0
        # ``sphinx-build -j`` forks reader processes. A forked process must not
        # talk to a worker started by its parent, so remember the owner.
        self.owner_pid = os.getpid()

    def render(self, job: _Job, timeout: float) -> RenderResult:
        self.conn.send(job)
        if not self.conn.poll(timeout):
            self._dump_and_kill()
            raise RenderError(
                f"Rendering example {job.clsname} timed out after {timeout:.0f}s; "
                "the render worker was killed (see stderr for its thread stacks)."
            )
        try:
            status, payload, rss_mb = self.conn.recv()
        except EOFError:
            self.process.join(5)
            raise RenderError(
                f"The render worker died while rendering example {job.clsname} "
                f"(exit code {self.process.exitcode}; -9 usually means it was "
                "killed for using too much memory)."
            ) from None
        self.renders += 1
        self.rss_mb = rss_mb
        if status != "ok":
            raise RenderError(
                f"Error while rendering example {job.clsname}:\n{payload}"
            )
        return payload

    def _dump_and_kill(self) -> None:
        if self.process.is_alive() and hasattr(signal, "SIGABRT"):
            # faulthandler in the worker prints all thread stacks on SIGABRT.
            os.kill(self.process.pid, signal.SIGABRT)
            self.process.join(10)
        if self.process.is_alive():
            self.process.kill()
        self.process.join(5)

    def stop(self) -> None:
        with contextlib.suppress(BrokenPipeError, OSError):
            self.conn.send(None)
        self.process.join(10)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(5)
        self.conn.close()


_worker: _Worker | None = None


def render_example(
    clsname: str,
    code: str,
    *,
    global_config: dict[str, Any],
    example_config: dict[str, Any],
) -> RenderResult:
    """Render a single example, isolated in a worker process.

    ``global_config`` is applied to the manim config of the worker before
    rendering (media paths, verbosity, ...); ``example_config`` is applied for
    the duration of this render only.
    """
    job = _Job(clsname, code, global_config, example_config)
    if RENDER_IN_PROCESS:
        try:
            return _render(job)
        except Exception as exc:
            raise RenderError(f"Error while rendering example {clsname}") from exc

    global _worker
    if _worker is not None and _worker.owner_pid != os.getpid():
        # Inherited from the parent through fork; leave it to the parent.
        _worker = None
    if _worker is None or not _worker.process.is_alive():
        _worker = _Worker()
    try:
        return _worker.render(job, RENDER_TIMEOUT)
    except RenderError:
        # The worker may be in an unknown state; never reuse it after a failure.
        _worker.stop()
        _worker = None
        raise
    finally:
        if _worker is not None and (
            _worker.renders >= MAX_RENDERS_PER_WORKER
            or _worker.rss_mb >= MAX_WORKER_RSS_MB
        ):
            _worker.stop()
            _worker = None


def shutdown(*args: Any) -> None:
    """Stop the current worker, if any (connected to ``build-finished``)."""
    global _worker
    if _worker is not None and _worker.owner_pid == os.getpid():
        _worker.stop()
    _worker = None
