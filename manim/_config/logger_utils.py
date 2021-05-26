"""Utilities to create and set the logger.

Manim's logger can be accessed as ``manim.logger``, or as
``logging.getLogger("manim")``, once the library has been imported.  Manim also
exports a second object, ``console``, which should be used to print on screen
messages that need not be logged.

Both ``logger`` and ``console`` use the ``rich`` library to produce rich text
format.

"""
import configparser
import copy
import json
import logging
import os
import sys
import typing
from typing import TYPE_CHECKING

from rich import color, errors
from rich import print as printf
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

if TYPE_CHECKING:
    from manim._config.utils import ManimConfig
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
    parser: configparser.ConfigParser, verbosity: str
) -> typing.Tuple[logging.Logger, Console]:
    """Make the manim logger and console.

    Parameters
    ----------
    parser : :class:`configparser.ConfigParser`
        A parser containing any .cfg files in use.

    verbosity : :class:`str`
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

    # With rich 9.5.0+ we could pass stderr=True instead
    error_console = Console(theme=theme, file=sys.stderr)

    # set the rich handler
    RichHandler.KEYWORDS = HIGHLIGHTED_KEYWORDS
    rich_handler = RichHandler(
        console=console, show_time=parser.getboolean("log_timestamps")
    )

    # finally, the logger
    logger = logging.getLogger("manim")
    logger.addHandler(rich_handler)
    logger.setLevel(verbosity)

    return logger, console, error_console


def parse_theme(parser: configparser.ConfigParser) -> Theme:
    """Configure the rich style of logger and console output.

    Parameters
    ----------
    parser : :class:`configparser.ConfigParser`
        A parser containing any .cfg files in use.

    Returns
    -------
    :class:`rich.Theme`
        The rich theme to be used by the manim logger.

    See Also
    --------
    :func:`make_logger`.

    """
    theme = {key.replace("_", "."): parser[key] for key in parser}

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
            }
        )
    except (color.ColorParseError, errors.StyleSyntaxError):
        printf(WRONG_COLOR_CONFIG_MSG)
        custom_theme = None

    return custom_theme


def set_file_logger(config: "ManimConfig", verbosity: str) -> None:
    """Add a file handler to manim logger.

    The path to the file is built using ``config.log_dir``.

    Parameters
    ----------
    config : :class:`ManimConfig`
        The global config, used to determine the log file path.

    verbosity : :class:`str`
        The verbosity level of the logger.

    Notes
    -----
    Calling this function changes the verbosity of all handlers assigned to
    manim logger.

    """
    # Note: The log file name will be
    # <name_of_animation_file>_<name_of_scene>.log, gotten from config.  So it
    # can differ from the real name of the scene.  <name_of_scene> would only
    # appear if scene name was provided when manim was called.
    scene_name_suffix = "".join(config["scene_names"])
    scene_file_name = os.path.basename(config["input_file"]).split(".")[0]
    log_file_name = (
        f"{scene_file_name}_{scene_name_suffix}.log"
        if scene_name_suffix
        else f"{scene_file_name}.log"
    )
    log_file_path = config.get_dir("log_dir") / log_file_name
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(JSONFormatter())

    logger = logging.getLogger("manim")
    logger.addHandler(file_handler)
    logger.info("Log file will be saved in %(logpath)s", {"logpath": log_file_path})

    config.verbosity = verbosity
    logger.setLevel(verbosity)


class JSONFormatter(logging.Formatter):
    """A formatter that outputs logs in a custom JSON format.

    This class is used internally for testing purposes.

    """

    def format(self, record: dict) -> str:
        """Format the record in a custom JSON format."""
        record_c = copy.deepcopy(record)
        if record_c.args:
            for arg in record_c.args:
                record_c.args[arg] = "<>"
        return json.dumps(
            {
                "levelname": record_c.levelname,
                "module": record_c.module,
                "message": super().format(record_c),
            }
        )
