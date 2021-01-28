import os
import click

from manim.constants import EPILOG
from manim.constants import HELP_OPTIONS

from manim._config import cfg_subcmds

__all__ = ["cfg", "write", "show", "export"]


@click.group(
    context_settings=HELP_OPTIONS,
    invoke_without_command=True,
    no_args_is_help=True,
    epilog=EPILOG,
    help="Manages Manim configuration files.",
)
def cfg():
    """Responsible for the cfg subcommand."""
    pass


@cfg.command(context_settings=HELP_OPTIONS, no_args_is_help=True)
@click.option(
    "-l",
    "--level",
    type=click.Choice(["user", "cwd"], case_sensitive=False),
    default="cwd",
    help="Specify if this config is for user or the working directory.",
)
@click.option("-o", "--open", is_flag=True)
def write(level, open):
    click.echo("write")


@cfg.command(context_settings=HELP_OPTIONS)
def show():
    click.echo("show")
    cfg_subcmds.show()


@cfg.command(context_settings=HELP_OPTIONS)
@click.option("-d", "--dir", default=os.getcwd())
def export(dir):
    click.echo("export")
    cfg_subcmds.export(dir)
