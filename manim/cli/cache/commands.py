"""Commands for maintaining cached video segments."""

from __future__ import annotations

from pathlib import Path

import cloup

from manim._config import config, console
from manim._config.output_plan import (
    resolve_module_name,
    resolve_segment_cache_directory,
)
from manim._config.utils import _determine_quality
from manim.cli.render.render_options import validate_resolution
from manim.constants import EPILOG, QUALITIES
from manim.utils.caching import clear_segment_cache

__all__ = ["cache"]


@cloup.group(
    context_settings=None,
    no_args_is_help=True,
    epilog=EPILOG,
)
def cache() -> None:
    """Maintain cached video segments."""


@cache.command(
    context_settings=None,
    no_args_is_help=True,
    epilog=EPILOG,
)
@cloup.argument(
    "file",
    type=cloup.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@cloup.argument("scene_names", required=True, nargs=-1)
@cloup.option(
    "-c",
    "--config-file",
    type=cloup.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Use the specified configuration file.",
)
@cloup.option(
    "--media-dir",
    type=cloup.Path(path_type=Path),
    default=None,
    help="Override the directory containing rendered media and caches.",
)
@cloup.option(
    "-q",
    "--quality",
    type=cloup.Choice(
        list(reversed([q["flag"] for q in QUALITIES.values() if q["flag"]])),
        case_sensitive=False,
    ),
    default=None,
    help="Resolve the cache directory for this render quality.",
)
@cloup.option(
    "-r",
    "--resolution",
    callback=validate_resolution,
    default=None,
    help='Resolve the cache directory for resolution "W,H".',
)
@cloup.option(
    "--fps",
    "--frame-rate",
    "frame_rate",
    type=float,
    default=None,
    help="Resolve the cache directory for this frame rate.",
)
def clear(
    *,
    file: Path,
    scene_names: tuple[str, ...],
    config_file: Path | None,
    media_dir: Path | None,
    quality: str | None,
    resolution: tuple[int, int] | None,
    frame_rate: float | None,
) -> None:
    """Delete cached segments for SCENE(S) from FILE."""
    command_config = config.copy()
    selected_config_file = file if file.suffix == ".cfg" else config_file
    if selected_config_file is not None:
        command_config.digest_file(selected_config_file)

    if not command_config.input_file:
        if file.suffix == ".cfg":
            raise ValueError("A configuration file must define input_file.")
        command_config.input_file = file.absolute()

    if media_dir is not None:
        command_config.media_dir = media_dir
    if quality is not None:
        command_config.quality = _determine_quality(quality)
    if resolution is not None:
        command_config.frame_size = resolution
    if frame_rate is not None:
        command_config.frame_rate = frame_rate

    module_name = resolve_module_name(command_config)
    working_directory = Path.cwd()
    for scene_name in scene_names:
        directory = resolve_segment_cache_directory(
            command_config,
            module_name=module_name,
            scene_name=scene_name,
            working_directory=working_directory,
        )
        removed = clear_segment_cache(directory)
        console.print(
            f"Removed {removed} cached segment(s) for {scene_name} from {directory}.",
        )
