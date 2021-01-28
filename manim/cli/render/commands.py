import click

from manim.constants import EPILOG
from manim.constants import HELP_OPTIONS


@click.command(
    context_settings=HELP_OPTIONS, help="Renders scenes from the file.", epilog=EPILOG
)
@click.argument("file", required=False)
@click.argument("scenes", required=False, nargs=-1)
def render(file, scenes):
    click.echo("render")
    print(file, scenes)
