from __future__ import annotations

import click
import cloup

from . import __version__, cli_ctx_settings, console
from .cli.cfg.group import cfg
from .cli.checkhealth.commands import checkhealth
from .cli.default_group import DefaultGroup
from .cli.init.commands import init
from .cli.plugins.commands import plugins
from .cli.render.commands import render
from .constants import EPILOG


def show_splash(ctx, param, value):
    if value:
        console.print(f"Manim Community [green]v{__version__}[/green]\n")


def print_version_and_exit(ctx, param, value):
    show_splash(ctx, param, value)
    if value:
        ctx.exit()


@cloup.group(
    context_settings=cli_ctx_settings,
    cls=DefaultGroup,
    default="render",
    no_args_is_help=True,
    help="Animation engine for explanatory math videos.",
    epilog="See 'manim <command>' to read about a specific subcommand.\n\n"
    "Note: the subcommand 'manim render' is called if no other subcommand "
    "is specified. Run 'manim render --help' if you would like to know what the "
    f"'-ql' or '-p' flags do, for example.\n\n{EPILOG}",
)
@cloup.option(
    "--version",
    is_flag=True,
    help="Show version and exit.",
    callback=print_version_and_exit,
    is_eager=True,
    expose_value=False,
)
@click.option(
    "--show-splash/--hide-splash",
    is_flag=True,
    default=True,
    help="Print splash message with version information.",
    callback=show_splash,
    is_eager=True,
    expose_value=False,
)
@cloup.pass_context
def main(ctx):
    """The entry point for manim."""
    pass


main.add_command(checkhealth)
main.add_command(cfg)
main.add_command(plugins)
main.add_command(init)
main.add_command(render)

if __name__ == "__main__":
    main()
