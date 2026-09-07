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
from manim.constants import EPILOG
from manim.timeline import _SourceSnapshot
from manim.utils.module_ops import (
    get_module,
    get_scene_classes_from_module,
    scene_classes_from_file,
)

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


def _validate_scene_batch_output_name(scene_classes: list[type]) -> None:
    if config.output_file and (config.write_all or len(scene_classes) != 1):
        raise ValueError(
            "--output_file can only be used when rendering exactly one scene. "
            "Remove --write_all or select a single scene.",
        )


@cloup.command(
    context_settings=None,
    no_args_is_help=True,
    epilog=EPILOG,
)
@cloup.argument("file", type=cloup.Path(path_type=Path), required=True)
@cloup.argument("scene_names", required=False, nargs=-1)
@cloup.option(
    "--timeline-output",
    type=cloup.Path(path_type=Path, dir_okay=False),
    help="Evaluate one scene without rendering and atomically save timeline JSON.",
)
@global_options
@output_options
@render_options
@ease_of_access_options
def render(**kwargs: Any) -> ClickArgs | dict[str, Any]:
    """Render SCENE(S) from the input FILE.

    FILE is the file path of the script or a config file.

    SCENES is an optional list of scenes in the file.
    """
    timeline_output = kwargs.pop("timeline_output", None)
    if timeline_output is not None:
        # User module/setup/construct code may change the process working directory.
        timeline_output = timeline_output.absolute()
    if timeline_output is not None and kwargs["jupyter"]:
        raise cloup.UsageError("Use Manager.evaluate(capture_timeline=True) in Python.")
    click_args = ClickArgs(kwargs)
    if kwargs["jupyter"]:
        return click_args

    config.digest_args(click_args)
    file = Path(config.input_file)
    try:
        if timeline_output is not None:
            if str(file) == "-":
                raise ValueError(
                    "--timeline-output requires a primary source file, not stdin."
                )
            if timeline_output.resolve() == file.resolve():
                raise ValueError(
                    "Timeline output must not replace the input source file."
                )
            if config.write_all or len(config.scene_names) > 1:
                raise ValueError(
                    "--timeline-output requires exactly one selected scene without --write_all."
                )
        source = (
            _SourceSnapshot.capture(
                file, "captured-before-loading-primary-bytes-compiled-directly"
            )
            if timeline_output is not None
            else None
        )
        if timeline_output is not None:
            assert source is not None
            if source.content is None:
                raise ValueError(
                    "--timeline-output requires a readable primary source file."
                )
            scene_classes = get_scene_classes_from_module(
                get_module(file, source=source.content)
            )
            requested = config.scene_names
            if requested:
                scene_classes = [
                    cls for cls in scene_classes if cls.__name__ == requested[0]
                ]
        else:
            scene_classes = scene_classes_from_file(file)
        _validate_scene_batch_output_name(scene_classes)

        if timeline_output is not None:
            if len(scene_classes) != 1 or config.write_all:
                raise ValueError(
                    "--timeline-output requires exactly one selected scene without --write_all."
                )
            assert source is not None
            if not source.unchanged():
                raise RuntimeError("Primary source changed while loading scenes.")
            with tempconfig({}):
                scene = scene_classes[0]()
                manager = scene._get_manager()
                with manager:
                    manager._timeline_source = source
                    manager.evaluate(capture_timeline=True)
                if not source.unchanged():
                    raise RuntimeError(
                        "Primary source changed during timeline resource cleanup."
                    )
                manager.timeline.write(timeline_output)
            return kwargs

        for SceneClass in scene_classes:
            while True:
                with tempconfig({}):
                    scene = SceneClass()
                    # Reuse a manager created by the scene's constructor. The
                    # with block also cleans up after custom render() overrides.
                    with scene._get_manager():
                        rerun = scene.render()
                if not rerun:
                    break
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
