from __future__ import annotations

import logging
import re
import sys
from typing import TYPE_CHECKING

from cloup import Choice, option, option_group

if TYPE_CHECKING:
    from click import Context, Option

__all__ = ["global_options"]

logger = logging.getLogger("manim")


def validate_gui_location(
    ctx: Context, param: Option, value: str | None
) -> tuple[int, int] | None:
    """If the ``value`` string is given, extract from it the GUI location,
    which should be in any of these formats: 'x;y', 'x,y' or 'x-y'.

    Parameters
    ----------
    ctx
        The Click context.
    param
        A Click option.
    value
        The optional string which will be parsed.

    Returns
    -------
    tuple[int, int] | None
        If ``value`` is ``None``, the return value is ``None``. Otherwise, it's
        the ``(x, y)`` location for the GUI.

    Raises
    ------
    ValueError
        If ``value`` has an invalid format.
    """
    if value is None:
        return None

    try:
        x_offset, y_offset = map(int, re.split(r"[;,\-]", value))
    except Exception:
        logger.error("GUI location option is invalid.")
        sys.exit()

    return (x_offset, y_offset)


global_options = option_group(
    "Global options",
    option(
        "-c",
        "--config_file",
        help="Specify the configuration file to use for render settings.",
        default=None,
    ),
    option(
        "--custom_folders",
        is_flag=True,
        default=None,
        help="Use the folders defined in the [custom_folders] section of the "
        "config file to define the output folder structure.",
    ),
    option(
        "--disable_caching",
        is_flag=True,
        default=None,
        help="Disable the use of the cache (still generates cache files).",
    ),
    option(
        "--flush_cache",
        is_flag=True,
        help="Remove cached partial movie files.",
        default=None,
    ),
    option("--tex_template", help="Specify a custom TeX template file.", default=None),
    option(
        "-v",
        "--verbosity",
        type=Choice(
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            case_sensitive=False,
        ),
        help="Verbosity of CLI output. Changes ffmpeg log level unless 5+.",
        default=None,
    ),
    option(
        "--notify_outdated_version/--silent",
        is_flag=True,
        default=None,
        help="Display warnings for outdated installation.",
    ),
    option(
        "--enable_gui",
        is_flag=True,
        help="Enable GUI interaction.",
        default=None,
    ),
    option(
        "--gui_location",
        default=None,
        callback=validate_gui_location,
        help="Starting location for the GUI.",
    ),
    option(
        "--fullscreen",
        is_flag=True,
        help="Expand the window to its maximum possible size.",
        default=None,
    ),
    option(
        "--enable_wireframe",
        is_flag=True,
        help="Enable wireframe debugging mode in opengl.",
        default=None,
    ),
    option(
        "--force_window",
        is_flag=True,
        help="Force window to open when using the opengl renderer, intended for debugging as it may impact performance",
        default=False,
    ),
    option(
        "--dry_run",
        is_flag=True,
        help="Renders animations without outputting image or video files and disables the window",
        default=False,
    ),
    option(
        "--no_latex_cleanup",
        is_flag=True,
        help="Prevents deletion of .aux, .dvi, and .log files produced by Tex and MathTex.",
        default=False,
    ),
    option(
        "--preview_command",
        help="The command used to preview the output file (for example vlc for video files)",
        default="",
    ),
)
