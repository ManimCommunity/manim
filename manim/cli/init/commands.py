"""Manim's init subcommand.

Manim's init subcommand is accessed in the command-line interface via ``manim
init``. Here you can specify options, subcommands, and subgroups for the init
group.

"""
from pathlib import Path

import click

from ... import console
from ...constants import CONTEXT_SETTINGS, EPILOG
from ...utils.file_ops import copy_template_files


@click.command(
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=False,
    epilog=EPILOG,
    help="Quickly setup a project",
)
def init():
    """The init subcommand is a quick and easy way to initialize a project
    It copies files from templates dir and pastes them in the current working dir
    """
    cfg = Path("manim.cfg")
    if cfg.exists():
        raise FileExistsError(f"\t{cfg} exists\n")
    else:
        copy_template_files()
