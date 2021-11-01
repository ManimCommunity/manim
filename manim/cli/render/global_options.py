import re

import click
from cloup import option, option_group

from ... import logger


def validate_gui_location(ctx, param, value):
    if value:
        try:
            x_offset, y_offset = map(int, re.split(r"[;,\-]", value))
            return (x_offset, y_offset)
        except Exception:
            logger.error("GUI location option is invalid.")
            exit()


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
        type=click.Choice(
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
)
