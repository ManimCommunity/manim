"""Auxiliary module for the checkhealth subcommand, contains
the actual check implementations."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Callable

from ..._config import config

HEALTH_CHECKS = []


def healthcheck(
    description: str,
    recommendation: str,
    skip_on_failed: list[Callable] | None = None,
    post_fail_fix_hook: Callable | None = None,
):
    """Decorator used for declaring health checks.

    This decorator attaches some data to a function,
    which is then added to a list containing all checks.

    Parameters
    ----------
    description
        A brief description of this check, displayed when
        the checkhealth subcommand is run.
    recommendation
        Help text which is displayed in case the check fails.
    skip_on_failed
        A list of check functions which, if they fail, cause
        the current check to be skipped.
    post_fail_fix_hook
        A function that is supposed to (interactively) help
        to fix the detected problem, if possible. This is
        only called upon explicit confirmation of the user.

    Returns
    -------
    A check function, as required by the checkhealth subcommand.
    """
    if skip_on_failed is None:
        skip_on_failed = []
    skip_on_failed = [
        skip.__name__ if callable(skip) else skip for skip in skip_on_failed
    ]

    def decorator(func):
        func.description = description
        func.recommendation = recommendation
        func.skip_on_failed = skip_on_failed
        func.post_fail_fix_hook = post_fail_fix_hook
        HEALTH_CHECKS.append(func)
        return func

    return decorator


@healthcheck(
    description="Checking whether manim is on your PATH",
    recommendation=(
        "The command <manim> is currently not on your system's PATH.\n\n"
        "You can work around this by calling the manim module directly "
        "via <python -m manim> instead of just <manim>.\n\n"
        "To fix the PATH issue properly: "
        "Usually, the Python package installer pip issues a warning "
        "during the installation which contains more information. "
        "Consider reinstalling manim via <pip uninstall manim> "
        "followed by <pip install manim> to see the warning again, "
        "then consult the internet on how to modify your system's "
        "PATH variable."
    ),
)
def is_manim_on_path():
    path_to_manim = shutil.which("manim")
    return path_to_manim is not None


@healthcheck(
    description="Checking whether the executable belongs to manim",
    recommendation="TODO",
    skip_on_failed=[is_manim_on_path],
)
def is_manim_executable_associated_to_this_library():
    path_to_manim = shutil.which("manim")
    with open(path_to_manim) as f:
        manim_exec = f.read()
    return "manim.__main__" in manim_exec


@healthcheck(
    description="Checking whether ffmpeg is available",
    recommendation="TODO",
)
def is_ffmpeg_available():
    path_to_ffmpeg = shutil.which(config.ffmpeg_executable)
    return path_to_ffmpeg is not None and os.access(path_to_ffmpeg, os.X_OK)


@healthcheck(
    description="Checking whether ffmpeg is working",
    recommendation="TODO",
    skip_on_failed=[is_ffmpeg_available],
)
def is_ffmpeg_working():
    ffmpeg_version = subprocess.run(
        [config.ffmpeg_executable, "-version"],
        stdout=subprocess.PIPE,
    ).stdout.decode()
    return (
        ffmpeg_version.startswith("ffmpeg version")
        and "--enable-libx264" in ffmpeg_version
    )


@healthcheck(
    description="Checking whether latex is available",
    recommendation="TODO",
)
def is_latex_available():
    path_to_latex = shutil.which("latex")
    return path_to_latex is not None and os.access(path_to_latex, os.X_OK)


@healthcheck(
    description="Checking whether dvisvgm is available",
    recommendation="TODO",
)
def is_dvisvgm_available():
    path_to_dvisvgm = shutil.which("dvisvgm")
    return path_to_dvisvgm is not None and os.access(path_to_dvisvgm, os.X_OK)
