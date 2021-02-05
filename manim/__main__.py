import sys
import click
from click.testing import CliRunner
from click_default_group import DefaultGroup
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
@click.version_option()
@click.pass_context
def main(ctx):
    """The entry point for manim."""
    pass


main.add_command(cfg)
main.add_command(plugins)
main.add_command(render)

if __name__ == "__main__":
    print(sys.argv)
    runner = CliRunner()
    result = runner.invoke(main, sys.argv)
    assert result.exit_code == 0
