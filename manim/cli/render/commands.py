"""Manim's default subcommand, render.

Manim's render subcommand is accessed in the command-line interface via
``manim``, but can be more explicitly accessed with ``manim render``. Here you
can specify options, and arguments for the render command.

"""
import sys
from pathlib import Path
from textwrap import dedent

import click
import cloup
import requests

from ... import __version__, config, console, logger
from ...constants import CONTEXT_SETTINGS, EPILOG
from ...utils.module_ops import scene_classes_from_file
from .ease_of_access_options import ease_of_access_options
from .global_options import global_options
from .output_options import output_options
from .render_options import render_options


@cloup.command(
    context_settings=CONTEXT_SETTINGS,
    epilog=EPILOG,
)
@click.argument("file", type=Path, required=False)
@click.argument("scene_names", required=False, nargs=-1)
@global_options
@output_options
@render_options
@ease_of_access_options
@click.pass_context
def render(
    ctx,
    **args,
):
    """Render SCENE(S) from the input FILE.

    FILE is the file path of the script.

    SCENES is an optional list of scenes in the file.
    """
    for scene in args["scene_names"]:
        if str(scene).startswith("-"):
            logger.warning(
                dedent(
                    """\
                Manim Community has moved to Click for the CLI.

                This means that options in the CLI are provided BEFORE the positional
                arguments for your FILE and SCENE(s):
                `manim render [OPTIONS] [FILE] [SCENES]...`

                For example:
                New way - `manim -p -ql file.py SceneName1 SceneName2 ...`
                Old way - `manim file.py SceneName1 SceneName2 ... -p -ql`

                To see the help page for the new available options, run:
                `manim render -h`
                """
                )
            )
            sys.exit()

    if args["use_opengl_renderer"]:
        logger.warning(
            "--use_opengl_renderer is deprecated, please use --renderer=opengl instead!"
        )
        renderer = "opengl"

    if args["use_webgl_renderer"]:
        logger.warning(
            "--use_webgl_renderer is deprecated, please use --renderer=webgl instead!"
        )
        renderer = "webgl"

    if args["use_webgl_renderer"] and args["use_opengl_renderer"]:
        logger.warning("You may select only one renderer!")
        sys.exit()

    class ClickArgs:
        def __init__(self, args):
            for name in args:
                setattr(self, name, args[name])

        def _get_kwargs(self):
            return list(self.__dict__.items())

        def __eq__(self, other):
            if not isinstance(other, ClickArgs):
                return NotImplemented
            return vars(self) == vars(other)

        def __contains__(self, key):
            return key in self.__dict__

        def __repr__(self):
            return str(self.__dict__)

    click_args = ClickArgs(args)
    if args["jupyter"]:
        return click_args

    nov = config.notify_outdated_version

    config.digest_args(click_args)
    file = args["file"]
    if config.renderer == "opengl":
        from manim.renderer.opengl_renderer import OpenGLRenderer

        for SceneClass in scene_classes_from_file(file):
            try:
                renderer = OpenGLRenderer()
                scene = SceneClass(renderer)
                scene.render()
            except Exception:
                console.print_exception()
    elif config.renderer == "webgl":
        try:
            from manim.grpc.impl import frame_server_impl

            server = frame_server_impl.get(file)
            server.start()
            server.wait_for_termination()
        except ModuleNotFoundError:
            console.print(
                "Dependencies for the WebGL render are missing. Run "
                "pip install manim[webgl_renderer] to install them."
            )
            console.print_exception()
    else:
        for SceneClass in scene_classes_from_file(file):
            try:
                scene = SceneClass()
                scene.render()
            except Exception:
                console.print_exception()

    if nov or args["notify_outdated_version"]:
        manim = requests.get("https://pypi.org/pypi/manim/json")
        releases = manim.json()["releases"].keys()
        stable = max(releases)

        if stable != __version__:
            console.print(
                f"You are using manim version [red]v{__version__}[/red], but version [green]v{stable}[/green] is available."
            )
            console.print(
                "You should consider upgrading via [yellow]pip installed -U manim[/yellow]"
            )

    return args
