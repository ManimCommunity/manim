r"""Render the examples of the ``manim`` directive ahead of the Sphinx build.

Rendering all examples inside one long ``sphinx-build`` call takes most of
the build time. To split that work into several shorter commands (for
example on CI services that struggle with long-running commands), the
build can be run in three steps:

1. Collect the examples, without rendering anything::

       cd docs/source
       sphinx-build -t manim-collect -b html -d /tmp/collect/doctrees . /tmp/collect/html

   With the ``manim-collect`` tag, the directive writes one job file per
   example to ``media/_prerender/jobs`` instead of rendering it.

2. Render the collected examples, in as many chunks as desired::

       python -m manim.utils.docbuild.prerender media/_prerender --chunk 1/4
       python -m manim.utils.docbuild.prerender media/_prerender --chunk 2/4
       ...

3. Build the documentation, using the rendered examples::

       sphinx-build -t manim-use-prerendered -b html ...

   With the ``manim-use-prerendered`` tag, the directive uses the result of
   a job whose example code and configuration match exactly, and renders
   the example itself otherwise (e.g. if it was not collected, or failed).

Results are only used with the ``manim-use-prerendered`` tag, so stale
results from an earlier run never end up in a regular build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from manim.utils.docbuild._render_worker import (
    RenderError,
    RenderResult,
    render_example,
    shutdown,
)

__all__ = ["job_key", "load_result", "main", "write_job"]


def job_key(
    clsname: str,
    code: str,
    global_config: dict[str, Any],
    example_config: dict[str, Any],
) -> str:
    """Identify an example by everything that affects its rendered output."""
    payload = json.dumps(
        {
            "clsname": clsname,
            "code": code,
            "global_config": global_config,
            "example_config": example_config,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def write_job(
    prerender_dir: Path,
    clsname: str,
    code: str,
    global_config: dict[str, Any],
    example_config: dict[str, Any],
) -> None:
    """Record an example to be rendered by :func:`main`."""
    key = job_key(clsname, code, global_config, example_config)
    jobs_dir = prerender_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "key": key,
        "clsname": clsname,
        "code": code,
        "global_config": global_config,
        "example_config": example_config,
        # Examples may refer to files relative to the Sphinx source directory.
        "cwd": os.getcwd(),
    }
    (jobs_dir / f"{key}.json").write_text(json.dumps(job, default=str, indent=1))


def load_result(
    prerender_dir: Path,
    clsname: str,
    code: str,
    global_config: dict[str, Any],
    example_config: dict[str, Any],
) -> RenderResult | None:
    """Return the prerendered result of an example, if there is a valid one."""
    key = job_key(clsname, code, global_config, example_config)
    result_file = prerender_dir / "results" / f"{key}.json"
    if not result_file.exists():
        return None
    data = json.loads(result_file.read_text())
    file_path = Path(data["file_path"])
    if not file_path.exists():
        return None
    return RenderResult(file_path=file_path, run_time=float(data["run_time"]))


def _parse_chunk(value: str) -> tuple[int, int]:
    index, _, total = value.partition("/")
    try:
        chunk = (int(index), int(total))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected I/N, got {value!r}") from None
    if not 1 <= chunk[0] <= chunk[1]:
        raise argparse.ArgumentTypeError(f"expected 1 <= I <= N, got {value!r}")
    return chunk


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m manim.utils.docbuild.prerender",
        description="Render examples collected with the manim-collect tag.",
    )
    parser.add_argument("prerender_dir", type=Path)
    parser.add_argument(
        "--chunk",
        type=_parse_chunk,
        default=(1, 1),
        metavar="I/N",
        help="render only the I-th of N chunks of the collected examples",
    )
    args = parser.parse_args(argv)
    prerender_dir: Path = args.prerender_dir.absolute()
    index, total = args.chunk

    job_files = sorted((prerender_dir / "jobs").glob("*.json"))
    # Interleave, so every chunk gets a mix of short and long examples.
    selected = job_files[index - 1 :: total]
    results_dir = prerender_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"chunk {index}/{total}: {len(selected)} of {len(job_files)} examples",
        flush=True,
    )

    failed = 0
    start = time.perf_counter()
    try:
        for number, job_file in enumerate(selected, 1):
            job = json.loads(job_file.read_text())
            result_file = results_dir / f"{job['key']}.json"
            if result_file.exists():
                continue
            # Set before the first render, so the worker starts in it as well.
            os.chdir(job["cwd"])
            label = f"[{number}/{len(selected)}] {job['clsname']}"
            try:
                result = render_example(
                    job["clsname"],
                    job["code"],
                    global_config=job["global_config"],
                    example_config=job["example_config"],
                )
            except RenderError as error:
                failed += 1
                # The final build renders this example again and reports it.
                print(f"{label}: FAILED\n{error}", flush=True)
                continue
            result_file.write_text(
                json.dumps(
                    {"file_path": str(result.file_path), "run_time": result.run_time}
                )
            )
            print(f"{label}: {result.run_time:.2f}s", flush=True)
    finally:
        shutdown()

    print(
        f"chunk {index}/{total} done in {time.perf_counter() - start:.0f}s, "
        f"{failed} failed",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
