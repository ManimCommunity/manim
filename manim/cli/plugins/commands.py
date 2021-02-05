import click

from manim.constants import EPILOG
from manim.constants import CONTEXT_SETTINGS

from manim.plugins.plugins_flags import list_plugins


@click.command(
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True,
    epilog=EPILOG,
    help="Manages Manim plugins.",
)
@click.option("-l", "--list", is_flag=True, help="List available plugins")
def plugins(list):
    click.echo("plugin")
    if list:
        list_plugins()
