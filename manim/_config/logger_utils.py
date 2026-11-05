"""Utilities to create and set the logger.

Manim's logger can be accessed as ``manim.logger``, or as
``logging.getLogger("manim")``, once the library has been imported.  Manim also
exports a second object, ``console``, which should be used to print on screen
messages that need not be logged.

Both ``logger`` and ``console`` use the ``rich`` library to produce rich text
format.

"""

from __future__ import annotations

import configparser
import copy
import json
import logging
import os
import sys
from typing import TYPE_CHECKING, Any

import decorator
from rich import color, errors
from rich import print as printf
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

if TYPE_CHECKING:
    from pathlib import Path
    from types import FrameType

__all__ = [
    "make_logger",
    "parse_theme",
    "set_file_logger",
    "JSONFormatter",
    "user_stacklevel",
]

# Frames from these locations are skipped by :func:`user_stacklevel`: the manim
# package itself, and the ``decorator`` library that wraps deprecated callables.
_INTERNAL_PATHS = (
    os.path.normcase(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    + os.sep,
    os.path.normcase(os.path.abspath(decorator.__file__)),
)

HIGHLIGHTED_KEYWORDS = [  # these keywords are highlighted specially
    "Played",
    "animations",
    "scene",
    "Reading",
    "Writing",
    "script",
    "arguments",
    "Invalid",
    "Aborting",
    "module",
    "File",
    "Rendering",
    "Rendered",
]

WRONG_COLOR_CONFIG_MSG = """
[logging.level.error]Your colour configuration couldn't be parsed.
Loading the default color configuration.[/logging.level.error]
"""


def make_logger(
    parser: configparser.SectionProxy,
    verbosity: str,
) -> tuple[logging.Logger, Console, Console]:
    """Make the manim logger and console.

    Parameters
    ----------
    parser
        A parser containing any .cfg files in use.

    verbosity
        The verbosity level of the logger.

    Returns
    -------
    :class:`logging.Logger`, :class:`rich.Console`, :class:`rich.Console`
        The manim logger and consoles. The first console outputs
        to stdout, the second to stderr. All use the theme returned by
        :func:`parse_theme`.

    See Also
    --------
    :func:`~._config.utils.make_config_parser`, :func:`parse_theme`

    Notes
    -----
    The ``parser`` is assumed to contain only the options related to
    configuring the logger at the top level.

    """
    # Throughout the codebase, use console.print() instead of print()
    theme = parse_theme(parser)
    console = Console(theme=theme)

    error_console = Console(theme=theme, stderr=True)

    # set the rich handler
    rich_handler = RichHandler(
        console=console,
        show_time=parser.getboolean("log_timestamps", fallback=False),
        keywords=HIGHLIGHTED_KEYWORDS,
    )

    # finally, the logger
    logger = logging.getLogger("manim")
    logger.addHandler(rich_handler)
    logger.setLevel(verbosity)
    logger.propagate = False

    if not (libav_logger := logging.getLogger()).hasHandlers():
        libav_logger.addHandler(rich_handler)
        libav_logger.setLevel(verbosity)

    return logger, console, error_console


def _is_internal_file(filename: str) -> bool:
    return os.path.normcase(os.path.abspath(filename)).startswith(_INTERNAL_PATHS)


def user_stacklevel() -> int:
    """Return the ``stacklevel`` that points a log record at the user's code.

    Warnings about something the user did wrong should tell them where in
    *their* code it happened. How many Manim frames lie between the user's
    line and the warning depends on the call path (``VGroup(a, a)`` goes
    through more frames than ``group.add(a, a)``), so a fixed ``stacklevel``
    does not work. Instead, this walks up the call stack and counts frames
    until it reaches the first one outside the ``manim`` package (frames of
    the ``decorator`` library, used for deprecations, are skipped as well).

    Call it directly inside the call to the logger, so that it is evaluated
    in the same frame as the logging call::

        logger.warning("...", stack_info=True, stacklevel=user_stacklevel())

    Returns
    -------
    int
        The ``stacklevel`` of the first frame outside of Manim, or ``1`` (the
        logging call itself) if every frame belongs to Manim.
    """
    frame: FrameType | None = sys._getframe(1)  # the function that is about to log
    stacklevel = 1
    while frame is not None and _is_internal_file(frame.f_code.co_filename):
        frame = frame.f_back
        stacklevel += 1
    return stacklevel if frame is not None else 1


def parse_theme(parser: configparser.SectionProxy) -> Theme | None:
    """Configure the rich style of logger and console output.

    Parameters
    ----------
    parser
        A parser containing any .cfg files in use.

    Returns
    -------
    :class:`rich.Theme`
        The rich theme to be used by the manim logger.

    See Also
    --------
    :func:`make_logger`.

    """
    theme: dict[str, Any] = {key.replace("_", "."): parser[key] for key in parser}

    theme["log.width"] = None if theme["log.width"] == "-1" else int(theme["log.width"])
    theme["log.height"] = (
        None if theme["log.height"] == "-1" else int(theme["log.height"])
    )
    theme["log.timestamps"] = False
    try:
        custom_theme = Theme(
            {
                k: v
                for k, v in theme.items()
                if k not in ["log.width", "log.height", "log.timestamps"]
            },
        )
    except (color.ColorParseError, errors.StyleSyntaxError):
        printf(WRONG_COLOR_CONFIG_MSG)
        custom_theme = None

    return custom_theme


def set_file_logger(log_file_path: Path) -> None:
    """Add a file handler for one exact, already resolved log path.

    Parameters
    ----------
    log_file_path
        Exact path of the log file for this scene.
    """
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(JSONFormatter())

    logger = logging.getLogger("manim")
    logger.addHandler(file_handler)
    logger.info("Log file will be saved in %(logpath)s", {"logpath": log_file_path})


class JSONFormatter(logging.Formatter):
    """A formatter that outputs logs in a custom JSON format.

    This class is used internally for testing purposes.

    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the record in a custom JSON format."""
        record_c = copy.deepcopy(record)
        if record_c.args:
            if isinstance(record_c.args, dict):
                for arg in record_c.args:
                    record_c.args[arg] = "<>"
            else:
                record_c.args = ("<>",) * len(record_c.args)
        return json.dumps(
            {
                "levelname": record_c.levelname,
                "module": record_c.module,
                "message": super().format(record_c),
            },
        )
