import sys

import click
from click_default_group import DefaultGroup

from . import __version__, console
from .cli.cfg.group import cfg
from .cli.init.commands import init
from .cli.new.group import new
from .cli.plugins.commands import plugins
from .cli.render.commands import render
from .constants import EPILOG


def exit_early(ctx, param, value):
    if value:
        sys.exit()


console.print(f"Manim Community [green]v{__version__}[/green]\n")


@click.group(
    cls=DefaultGroup,
    default="render",
    no_args_is_help=True,
    help="Animation engine for explanatory math videos",
    epilog=EPILOG,
)
@click.option(
    "--version",
    is_flag=True,
    help="Show version and exit.",
    callback=exit_early,
    is_eager=True,
    expose_value=False,
)
@click.pass_context
def main(ctx):
    """The entry point for manim."""
    pass


main.add_command(cfg)
main.add_command(plugins)
main.add_command(init)
main.add_command(new)
main.add_command(render)

if __name__ == "__main__":
    main()
