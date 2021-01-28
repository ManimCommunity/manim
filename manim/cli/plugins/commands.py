import click

from manim.constants import EPILOG
from manim.constants import HELP_OPTIONS

from manim.plugins.plugins_flags import list_plugins


@click.command(
    context_settings=HELP_OPTIONS,
    no_args_is_help=True,
    epilog=EPILOG,
    help="Manages Manim plugins.",
)
@click.option("-l", "--list", is_flag=True, help="List available plugins.")
def plugins(list):
    click.echo("plugins")
    if list:
        list_plugins()
