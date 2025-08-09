"""Manim's default subcommand, render.

Manim's render subcommand is accessed in the command-line interface via
``manim``, but can be more explicitly accessed with ``manim render``. Here you
can specify options, and arguments for the render command.

"""

from __future__ import annotations

import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import cloup

from manim import __version__
from manim._config import (
    config,
    console,
    error_console,
    logger,
    tempconfig,
)
from manim.cli.cli_utils import code_input_prompt, prompt_user_with_list
from manim.cli.render.ease_of_access_options import ease_of_access_options
from manim.cli.render.global_options import global_options
from manim.cli.render.output_options import output_options
from manim.cli.render.render_options import render_options
from manim.constants import (
    EPILOG,
    INVALID_NUMBER_MESSAGE,
    NO_SCENE_MESSAGE,
    SCENE_NOT_FOUND_MESSAGE,
    RendererType,
)
from manim.scene.scene_file_writer import SceneFileWriter
from manim.utils.module_ops import (
    module_from_file,
    module_from_text,
    search_classes_from_module,
)

__all__ = ["render"]

if TYPE_CHECKING:
    from ...scene.scene import Scene

INPUT_CODE_RENDER = "Rendering animation from typed code"
MULTIPLE_SCENES = "Found multiple scenes. Choose at least one to continue"


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
    warn_and_change_deprecated_arguments(kwargs)

    click_args = ClickArgs(kwargs)
    if kwargs["jupyter"]:
        return click_args

    config.digest_args(click_args)

    scenes = scenes_from_input(config.input_file)

    if config.renderer == RendererType.OPENGL:
        from manim.renderer.opengl_renderer import OpenGLRenderer

        try:
            renderer = OpenGLRenderer()
            keep_running = True
            while keep_running:
                for SceneClass in scenes:
                    with tempconfig({}):
                        scene = SceneClass(renderer)
                        rerun = scene.render()
                    if rerun or config.write_all:
                        renderer.num_plays = 0
                        continue
                    else:
                        keep_running = False
                        break
                if config.write_all:
                    keep_running = False

        except Exception:
            error_console.print_exception()
            sys.exit(1)
    else:
        for SceneClass in scenes:
            try:
                with tempconfig({}):
                    scene = SceneClass()
                    scene.render()
            except Exception:
                error_console.print_exception()
                sys.exit(1)

    if config.notify_outdated_version:
        version_notification()

    return kwargs


def version_notification() -> None:
    """Compare used version to latest version of manim.
    Version info is fetched from internet once a day and cached into a file.
    """
    stable_version = None

    cache_file = Path(os.path.dirname(__file__)) / ".version_cache.log"

    if cache_file.exists():
        with cache_file.open() as f:
            cache_lifetime = int(f.readline())
            if time.time() < cache_lifetime:
                stable_version = f.readline()

    if stable_version is None:
        version = fetch_version()
        if version is None:
            return None

        with cache_file.open(mode="w") as f:
            timecode = int(time.time()) + 86_400
            f.write(str(timecode) + "\n" + str(version))
        stable_version = version

    if stable_version != __version__:
        console.print(
            f"You are using manim version [red]v{__version__}[/red], but version [green]v{stable_version}[/green] is available.",
        )
        console.print(
            "You should consider upgrading via [yellow]pip install -U manim[/yellow]",
        )


def fetch_version() -> str | None:
    """Fetch latest manim version from PYPI-database"""
    import http.client
    import urllib.error
    import urllib.request

    manim_info_url = "https://pypi.org/pypi/manim/json"
    warn_prompt = "Cannot check if latest release of manim is installed"
    request = urllib.request.Request(manim_info_url)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response = cast(http.client.HTTPResponse, response)
            json_data = json.loads(response.read())

    except (Exception, urllib.error.HTTPError, urllib.error.URLError) as e:
        logger.debug(f"{e}: {warn_prompt} ")
        return None
    except json.JSONDecodeError:
        logger.debug(
            f"Error while decoding JSON from [{manim_info_url}]: {warn_prompt}"
        )
        return None
    else:
        return str(json_data["info"]["version"])


def warn_and_change_deprecated_arguments(kwargs: dict[str, Any]) -> None:
    """Helper function to print info about deprecated arguments
    and mutate inserted dictionary to use new format
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


def select_scenes(scene_classes: list[type[Scene]]) -> list[type[Scene]]:
    """Assortment of selection functionality in which one or more Scenes are selected from list.

    Parameters
    ----------
    scene_classes
        list of scene classes that
    """
    if config.write_all:
        return scene_classes

    result = []
    for scene_name in config.scene_names:
        found = False
        for scene_class in scene_classes:
            if scene_class.__name__ == scene_name:
                result.append(scene_class)
                found = True
                break
        if not found and (scene_name != ""):
            logger.error(SCENE_NOT_FOUND_MESSAGE.format(scene_name))
    if result:
        return result

    if len(scene_classes) == 1:
        config.scene_names = [scene_classes[0].__name__]
        return [scene_classes[0]]

    try:
        console.print(f"{MULTIPLE_SCENES}:\n", style="underline white")
        scene_indices = prompt_user_with_list([a.__name__ for a in scene_classes])
    except Exception as e:
        logger.error(f"{e}\n{INVALID_NUMBER_MESSAGE} ")
        sys.exit(2)

    classes = [scene_classes[i] for i in scene_indices]

    config.scene_names = [scene_class.__name__ for scene_class in classes]
    SceneFileWriter.force_output_as_scene_name = True

    return classes


def scenes_from_input(file_path_input: str) -> list[type[Scene]]:
    """Return scenes from file path or create CLI prompt for input

    Parameters
    ----------
    file_path_input
        file path or '-' that will open a code prompt
    """
    from ...scene.scene import Scene

    if file_path_input == "-":
        try:
            code = code_input_prompt()
            module = module_from_text(code)
        except Exception as e:
            logger.error(f"Failed to create from input code: {e}")
            sys.exit(2)

        logger.info(INPUT_CODE_RENDER)
    else:
        module = module_from_file(Path(file_path_input))

    try:
        scenes = search_classes_from_module(module, Scene)
        return select_scenes(scenes)
    except ValueError:
        logger.error(NO_SCENE_MESSAGE)
        return []
