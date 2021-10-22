"""Manim's init subcommand.

Manim's init subcommand is accessed in the command-line interface via ``manim
init``. Here you can specify options, subcommands, and subgroups for the init
group.

"""
from pathlib import Path

import click

from ...constants import CONTEXT_SETTINGS, EPILOG
from ...utils.file_ops import copy_template_files


@click.command(
    context_settings=CONTEXT_SETTINGS,
    epilog=EPILOG,
    short_help="""Sets up a new project in current working directory with default settings.\n
It copies files from templates directory and pastes them in the current working dir.
""",
)
def init():
    """Sets up a new project in current working directory with default settings.

    It copies files from templates directory and pastes them in the current working dir.
    """
    cfg = Path("manim.cfg")
    if cfg.exists():
        raise FileExistsError(f"\t{cfg} exists\n")
    else:
        copy_template_files()
