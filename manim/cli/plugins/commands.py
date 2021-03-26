"""Manim's plugin subcommand.

Manim's plugin subcommand is accessed in the command-line interface via ``manim
plugin``. Here you can specify options, subcommands, and subgroups for the plugin 
group.

"""
import click

from ...constants import EPILOG
from ...constants import CONTEXT_SETTINGS

from ...plugins.plugins_flags import list_plugins


@click.command(
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True,
    epilog=EPILOG,
    help="Manages Manim plugins.",
)
@click.option("-l", "--list", is_flag=True, help="List available plugins.")
def plugins(list):
    if list:
        list_plugins()
