import sys
import click
from click_default_group import DefaultGroup
from . import __version__, console
from .constants import EPILOG
from .constants import CONTEXT_SETTINGS
from .cli.cfg.commands import cfg
from .cli.plugins.commands import plugins
from .cli.render.commands import render


def exit_early(ctx, param, value):
    if value:
        sys.exit()


console.print(f"Manim Community [green]v{__version__}[/green]")


@click.group(
    cls=DefaultGroup,
    default="render",
    no_args_is_help=True,
    context_settings=CONTEXT_SETTINGS,
    help="Animation engine for explanatory math videos",
    epilog=EPILOG,
)
@click.option(
    "-v",
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
main.add_command(render)

if __name__ == "__main__":
    main()
