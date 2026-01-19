"""Manim's default subcommand, render.

Manim's render subcommand is accessed in the command-line interface via
``manim``, but can be more explicitly accessed with ``manim render``. Here you
can specify options, and arguments for the render command.

"""

from __future__ import annotations

import http.client
import json
import sys
import urllib.error
import urllib.request
from argparse import Namespace
from pathlib import Path
from typing import Any, cast

import cloup

from manim import __version__
from manim._config import (
    config,
    console,
    error_console,
    logger,
    tempconfig,
)
from manim.cli.render.ease_of_access_options import ease_of_access_options
from manim.cli.render.global_options import global_options
from manim.cli.render.output_options import output_options
from manim.cli.render.render_options import render_options
from manim.constants import EPILOG, RendererType
from manim.utils.module_ops import scene_classes_from_file

__all__ = ["render"]


class ClickArgs(Namespace):
    def __init__(self, args: dict[str, Any]) -> None:
        for name in args:
            setattr(self, name, args[name])

    def _get_kwargs(self) -> list[tuple[str, Any]]:
        return list(self.__dict__.items())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClickArgs):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def __repr__(self) -> str:
        return str(self.__dict__)


@cloup.command(
    context_settings=None,
    no_args_is_help=True,
    epilog=EPILOG,
)
@cloup.argument("file", type=cloup.Path(path_type=Path), required=True)
@cloup.argument("scene_names", required=False, nargs=-1)
@global_options
@output_options
@render_options
@ease_of_access_options
def render(**kwargs: Any) -> ClickArgs | dict[str, Any]:
    """Render SCENE(S) from the input FILE.

    FILE is the file path of the script or a config file.

    SCENES is an optional list of scenes in the file.
    """
    if kwargs["save_as_gif"]:
        logger.warning("--save_as_gif is deprecated, please use --format=gif instead!")
        kwargs["format"] = "gif"

    if kwargs["save_pngs"]:
        logger.warning("--save_pngs is deprecated, please use --format=png instead!")
        kwargs["format"] = "png"

    if kwargs["show_in_file_browser"]:
        logger.warning(
            "The short form of show_in_file_browser is deprecated and will be moved to support --format.",
        )

    click_args = ClickArgs(kwargs)
    if kwargs["jupyter"]:
        return click_args

    config.digest_args(click_args)
    file = Path(config.input_file)
    if config.renderer == RendererType.OPENGL:
        from manim.renderer.opengl_renderer import OpenGLRenderer

        try:
            renderer = OpenGLRenderer()
            keep_running = True
            while keep_running:
                for SceneClass in scene_classes_from_file(file):
                    with tempconfig({}):
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
    else:
        for SceneClass in scene_classes_from_file(file):
            try:
                with tempconfig({}):
                    scene = SceneClass()
                    scene.render()
            except Exception:
                error_console.print_exception()
                sys.exit(1)

    if config.notify_outdated_version:
        manim_info_url = "https://pypi.org/pypi/manim/json"
        warn_prompt = "Cannot check if latest release of manim is installed"

        try:
            with urllib.request.urlopen(
                urllib.request.Request(manim_info_url),
                timeout=10,
            ) as response:
                response = cast(http.client.HTTPResponse, response)
                json_data = json.loads(response.read())
        except urllib.error.HTTPError:
            logger.debug("HTTP Error: %s", warn_prompt)
        except urllib.error.URLError:
            logger.debug("URL Error: %s", warn_prompt)
        except json.JSONDecodeError:
            logger.debug(
                "Error while decoding JSON from %r: %s", manim_info_url, warn_prompt
            )
        except Exception:
            logger.debug("Something went wrong: %s", warn_prompt)
        else:
            stable = json_data["info"]["version"]
            if stable != __version__:
                console.print(
                    f"You are using manim version [red]v{__version__}[/red], but version [green]v{stable}[/green] is available.",
                )
                console.print(
                    "You should consider upgrading via [yellow]pip install -U manim[/yellow]",
                )

    return kwargs
