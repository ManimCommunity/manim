import click
from click_default_group import DefaultGroup
from manim import __version__
from manim.constants import EPILOG
from manim.constants import CONTEXT_SETTINGS
from manim.cli.cfg.commands import cfg
from manim.cli.plugins.commands import plugins
from manim.cli.render.commands import render


@click.group(
    cls=DefaultGroup,
    default="render",
    no_args_is_help=True,
    context_settings=CONTEXT_SETTINGS,
    help="Animation engine for explanatory math videos",
    epilog=EPILOG,
)
@click.version_option(
    version=__version__, prog_name="Manim", message="%(prog)s v%(version)s"
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
