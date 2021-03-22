import click
from click_default_group import DefaultGroup
from . import __version__
from .constants import EPILOG
from .constants import CONTEXT_SETTINGS
from .cli.cfg.commands import cfg
from .cli.plugins.commands import plugins
from .cli.render.commands import render


@click.group(
    cls=DefaultGroup,
    default="render",
    no_args_is_help=True,
    context_settings=CONTEXT_SETTINGS,
    help="Animation engine for explanatory math videos",
    epilog=EPILOG,
)
@click.version_option(
    version=__version__, prog_name="Manim Community", message="%(prog)s v%(version)s"
)
@click.pass_context
def manim(ctx):
    """The entry point for manim."""
    pass


manim.add_command(cfg)
manim.add_command(plugins)
manim.add_command(render)

if __name__ == "__main__":
    manim()
