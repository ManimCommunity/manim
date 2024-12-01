"""Auxiliary module for the checkhealth subcommand, contains
the actual check implementations.
"""

from __future__ import annotations

import os
import shutil
from typing import Callable, Protocol, cast

__all__ = ["HEALTH_CHECKS"]


class HealthCheckFunction(Protocol):
    description: str
    recommendation: str
    skip_on_failed: list[str]
    post_fail_fix_hook: Callable[..., object] | None
    __name__: str

    def __call__(self) -> bool: ...


HEALTH_CHECKS: list[HealthCheckFunction] = []


def healthcheck(
    description: str,
    recommendation: str,
    skip_on_failed: list[HealthCheckFunction | str] | None = None,
    post_fail_fix_hook: Callable[..., object] | None = None,
) -> Callable[[Callable[[], bool]], HealthCheckFunction]:
    """Decorator used for declaring health checks.

    This decorator attaches some data to a function, which is then added to a
    a list containing all checks.

    Parameters
    ----------
    description
        A brief description of this check, displayed when the ``checkhealth``
        subcommand is run.
    recommendation
        Help text which is displayed in case the check fails.
    skip_on_failed
        A list of check functions which, if they fail, cause the current check
        to be skipped.
    post_fail_fix_hook
        A function that is meant to (interactively) help to fix the detected
        problem, if possible. This is only called upon explicit confirmation of
        the user.

    Returns
    -------
    Callable[Callable[[], bool], :class:`HealthCheckFunction`]
        A decorator which converts a function into a health check function, as
        required by the ``checkhealth`` subcommand.
    """
    new_skip_on_failed: list[str]
    if skip_on_failed is None:
        new_skip_on_failed = []
    else:
        new_skip_on_failed = [
            skip.__name__ if callable(skip) else skip for skip in skip_on_failed
        ]

    def wrapper(func: Callable[[], bool]) -> HealthCheckFunction:
        health_func = cast(HealthCheckFunction, func)
        health_func.description = description
        health_func.recommendation = recommendation
        health_func.skip_on_failed = new_skip_on_failed
        health_func.post_fail_fix_hook = post_fail_fix_hook
        HEALTH_CHECKS.append(health_func)
        return health_func

    return wrapper


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
def is_manim_on_path() -> bool:
    """Check whether ``manim`` is in ``PATH``.

    Returns
    -------
    :class:`bool`
        Whether ``manim`` is in ``PATH`` or not.
    """
    path_to_manim = shutil.which("manim")
    return path_to_manim is not None


@healthcheck(
    description="Checking whether the executable belongs to manim",
    recommendation=(
        "The command <manim> does not belong to your installed version "
        "of this library, it likely belongs to manimgl / manimlib.\n\n"
        "Run manim via <python -m manim> or via <manimce>, or uninstall "
        "and reinstall manim via <pip install --upgrade "
        "--force-reinstall manim> to fix this."
    ),
    skip_on_failed=[is_manim_on_path],
)
def is_manim_executable_associated_to_this_library() -> bool:
    """Check whether the ``manim`` executable in ``PATH`` is associated to this
    library. To verify this, the executable should look like this:

    .. code-block:: python

        #!<MANIM_PATH>/.../python
        import sys
        from manim.__main__ import main

        if __name__ == "__main__":
            sys.exit(main())


    Returns
    -------
    :class:`bool`
        Whether the ``manim`` executable in ``PATH`` is associated to this
        library or not.
    """
    path_to_manim = shutil.which("manim")
    assert path_to_manim is not None
    with open(path_to_manim, "rb") as manim_binary:
        manim_exec = manim_binary.read()

    # first condition below corresponds to the executable being
    # some sort of python script. second condition happens when
    # the executable is actually a Windows batch file.
    return b"manim.__main__" in manim_exec or b'"%~dp0\\manim"' in manim_exec


@healthcheck(
    description="Checking whether latex is available",
    recommendation=(
        "Manim cannot find <latex> on your system's PATH. "
        "You will not be able to use Tex and MathTex mobjects "
        "in your scenes.\n\n"
        "Consult our installation instructions "
        "at https://docs.manim.community/en/stable/installation.html "
        "or search the web for instructions on how to install a "
        "LaTeX distribution on your operating system."
    ),
)
def is_latex_available() -> bool:
    """Check whether ``latex`` is in ``PATH`` and can be executed.

    Returns
    -------
    :class:`bool`
        Whether ``latex`` is in ``PATH`` and can be executed or not.
    """
    path_to_latex = shutil.which("latex")
    return path_to_latex is not None and os.access(path_to_latex, os.X_OK)


@healthcheck(
    description="Checking whether dvisvgm is available",
    recommendation=(
        "Manim could find <latex>, but not <dvisvgm> on your system's "
        "PATH. Make sure your installed LaTeX distribution comes with "
        "dvisvgm and consider installing a larger distribution if it "
        "does not."
    ),
    skip_on_failed=[is_latex_available],
)
def is_dvisvgm_available() -> bool:
    """Check whether ``dvisvgm`` is in ``PATH`` and can be executed.

    Returns
    -------
    :class:`bool`
        Whether ``dvisvgm`` is in ``PATH`` and can be executed or not.
    """
    path_to_dvisvgm = shutil.which("dvisvgm")
    return path_to_dvisvgm is not None and os.access(path_to_dvisvgm, os.X_OK)
