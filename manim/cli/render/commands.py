"""Manim's default subcommand, render.

Manim's render subcommand is accessed in the command-line interface via
``manim``, but can be more explicitly accessed with ``manim render``. Here you
can specify options, and arguments for the render command.

"""
import json
import sys
from pathlib import Path

import click
import cloup
import requests

from ... import __version__, config, console, error_console, logger
from ...constants import EPILOG
from ...utils.module_ops import scene_classes_from_file
from .ease_of_access_options import ease_of_access_options
from .global_options import global_options
from .output_options import output_options
from .render_options import render_options


@cloup.command(
    context_settings=None,
    epilog=EPILOG,
)
@click.argument("file", type=Path, required=True)
@click.argument("scene_names", required=False, nargs=-1)
@global_options
@output_options
@render_options
@ease_of_access_options
def render(
    **args,
):
    """Render SCENE(S) from the input FILE.

    FILE is the file path of the script.

    SCENES is an optional list of scenes in the file.
    """

    if args["use_opengl_renderer"]:
        logger.warning(
            "--use_opengl_renderer is deprecated, please use --renderer=opengl instead!",
        )
        args["renderer"] = "opengl"

    if args["use_webgl_renderer"]:
        logger.warning(
            "--use_webgl_renderer is deprecated, please use --renderer=webgl instead!",
        )
        args["renderer"] = "webgl"

    if args["use_webgl_renderer"] and args["use_opengl_renderer"]:
        logger.warning("You may select only one renderer!")
        sys.exit()

    if args["save_as_gif"]:
        logger.warning("--save_as_gif is deprecated, please use --format=gif instead!")
        args["format"] = "gif"

    if args["save_pngs"]:
        logger.warning("--save_pngs is deprecated, please use --format=png instead!")
        args["format"] = "png"

    if args["show_in_file_browser"]:
        logger.warning(
            "The short form of show_in_file_browser is deprecated and will be moved to support --format.",
        )

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

    config.digest_args(click_args)
    file = args["file"]
    if config.renderer == "opengl":
        from manim.renderer.opengl_renderer import OpenGLRenderer

        try:
            renderer = OpenGLRenderer()
            keep_running = True
            while keep_running:
                for SceneClass in scene_classes_from_file(file):
                    scene = SceneClass(renderer)
                    rerun = scene.render()
                    if rerun or config["write_all"]:
                        renderer.num_plays = 0
                        continue
                    else:
                        keep_running = False
                        break
                if config["write_all"]:
                    keep_running = False

        except Exception:
            error_console.print_exception()
            sys.exit(1)
    elif config.renderer == "webgl":
        try:
            from manim.grpc.impl import frame_server_impl

            server = frame_server_impl.get(file)
            server.start()
            server.wait_for_termination()
        except ModuleNotFoundError:
            console.print(
                "Dependencies for the WebGL render are missing. Run "
                "pip install manim[webgl_renderer] to install them.",
            )
            error_console.print_exception()
            sys.exit(1)
    else:
        for SceneClass in scene_classes_from_file(file):
            try:
                scene = SceneClass()
                scene.render()
            except Exception:
                error_console.print_exception()
                sys.exit(1)

    if config.notify_outdated_version:
        manim_info_url = "https://pypi.org/pypi/manim/json"
        warn_prompt = "Cannot check if latest release of manim is installed"
        req_info = {}

        try:
            req_info = requests.get(manim_info_url)
            req_info.raise_for_status()

            stable = req_info.json()["info"]["version"]
            if stable != __version__:
                console.print(
                    f"You are using manim version [red]v{__version__}[/red], but version [green]v{stable}[/green] is available.",
                )
                console.print(
                    "You should consider upgrading via [yellow]pip install -U manim[/yellow]",
                )
        except requests.exceptions.HTTPError:
            logger.debug(f"HTTP Error: {warn_prompt}")
        except requests.exceptions.ConnectionError:
            logger.debug(f"Connection Error: {warn_prompt}")
        except requests.exceptions.Timeout:
            logger.debug(f"Timed Out: {warn_prompt}")
        except json.JSONDecodeError:
            logger.debug(warn_prompt)
            logger.debug(f"Error decoding JSON from {manim_info_url}")
        except Exception:
            logger.debug(f"Something went wrong: {warn_prompt}")

    return args
