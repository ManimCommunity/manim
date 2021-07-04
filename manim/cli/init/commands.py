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
)
def init():
    """Sets up a project in current working directory with default settings.

    It copies files from templates directory and pastes them in the current working dir.

    The new project is set up with default settings.
    """
    cfg = Path("manim.cfg")
    if cfg.exists():
        raise FileExistsError(f"\t{cfg} exists\n")
    else:
        copy_template_files()
