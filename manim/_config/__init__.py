"""Set the global config and logger."""

from __future__ import annotations

import logging
from contextlib import _GeneratorContextManager, contextmanager

from .cli_colors import parse_cli_ctx
from .logger_utils import make_logger
from .utils import ManimConfig, ManimFrame, make_config_parser

__all__ = [
    "logger",
    "console",
    "error_console",
    "config",
    "frame",
    "tempconfig",
    "cli_ctx_settings",
]

parser = make_config_parser()
logger: logging.Logger

# The logger can be accessed from anywhere as manim.logger, or as
# logging.getLogger("manim").  The console must be accessed as manim.console.
# Throughout the codebase, use manim.console.print() instead of print().
# Use error_console to print errors so that it outputs to stderr.
logger, console, error_console = make_logger(
    parser["logger"],
    parser["CLI"]["verbosity"],
)
cli_ctx_settings = parse_cli_ctx(parser["CLI_CTX"])
# TODO: temporary to have a clean terminal output when working with PIL or matplotlib
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.INFO)

config = ManimConfig().digest_parser(parser)
frame = ManimFrame(config)


# This has to go here because it needs access to this module's config
@contextmanager
def tempconfig(temp: ManimConfig | dict) -> _GeneratorContextManager:
    """Context manager that temporarily modifies the global ``config`` object.

    Inside the ``with`` statement, the modified config will be used.  After
    context manager exits, the config will be restored to its original state.

    Parameters
    ----------
    temp
        Object whose keys will be used to temporarily update the global
        ``config``.

    Examples
    --------

    Use ``with tempconfig({...})`` to temporarily change the default values of
    certain config options.

    .. code-block:: pycon

       >>> config["frame_height"]
       8.0
       >>> with tempconfig({"frame_height": 100.0}):
       ...     print(config["frame_height"])
       ...
       100.0
       >>> config["frame_height"]
       8.0

    """
    global config
    original = config.copy()

    temp = {k: v for k, v in temp.items() if k in original}

    # In order to change the config that every module has access to, use
    # update(), DO NOT use assignment.  Assigning config = some_dict will just
    # make the local variable named config point to a new dictionary, it will
    # NOT change the dictionary that every module has a reference to.
    config.update(temp)
    try:
        yield
    finally:
        config.update(original)  # update, not assignment!
