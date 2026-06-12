from __future__ import annotations

import logging
import re
import sys
from typing import TYPE_CHECKING

from cloup import Choice, option, option_group

from manim.constants import QUALITIES, RendererType

if TYPE_CHECKING:
    from click import Context, Option

__all__ = ["render_options"]

logger = logging.getLogger("manim")


def validate_scene_range(
    ctx: Context, param: Option, value: str | None
) -> tuple[int] | tuple[int, int] | None:
    """If the ``value`` string is given, extract from it the scene range, which
    should be in any of these formats: 'start', 'start;end', 'start,end' or
    'start-end'. Otherwise, return ``None``.

    Parameters
    ----------
    ctx
        The Click context.
    param
        A Click option.
    value
        The optional string which will be parsed.

    Returns
    -------
    tuple[int] | tuple[int, int] | None
        If ``value`` is ``None``, the return value is ``None``. Otherwise, it's
        the scene range, given by a tuple which may contain a single value
        ``start`` or two values ``start`` and ``end``.

    Raises
    ------
    ValueError
        If ``value`` has an invalid format.
    """
    if value is None:
        return None

    try:
        start = int(value)
        return (start,)
    except Exception:
        pass

    try:
        start, end = map(int, re.split(r"[;,\-]", value))
    except Exception:
        logger.error("Couldn't determine a range for -n option.")
        sys.exit()

    return start, end


def validate_resolution(
    ctx: Context, param: Option, value: str | None
) -> tuple[int, int] | None:
    """If the ``value`` string is given, extract from it the resolution, which
    should be in any of these formats: 'W;H', 'W,H' or 'W-H'. Otherwise, return
    ``None``.

    Parameters
    ----------
    ctx
        The Click context.
    param
        A Click option.
    value
        The optional string which will be parsed.

    Returns
    -------
    tuple[int, int] | None
        If ``value`` is ``None``, the return value is ``None``. Otherwise, it's
        the resolution as a ``(W, H)`` tuple.

    Raises
    ------
    ValueError
        If ``value`` has an invalid format.
    """
    if value is None:
        return None

    try:
        width, height = map(int, re.split(r"[;,\-]", value))
    except Exception:
        logger.error("Resolution option is invalid.")
        sys.exit()

    return width, height


render_options = option_group(
    "Render Options",
    option(
        "-n",
        "--from_animation_number",
        callback=validate_scene_range,
        help="Start rendering from n_0 until n_1. If n_1 is left unspecified, "
        "renders all scenes after n_0.",
        default=None,
    ),
    option(
        "-a",
        "--write_all",
        is_flag=True,
        help="Render all scenes in the input file.",
        default=None,
    ),
    option(
        "--format",
        type=Choice(["png", "gif", "mp4", "webm", "mov"], case_sensitive=False),
        default=None,
    ),
    option(
        "-s",
        "--save_last_frame",
        default=None,
        is_flag=True,
        help="Render and save only the last frame of a scene as a PNG image.",
    ),
    option(
        "-q",
        "--quality",
        default=None,
        type=Choice(
            list(reversed([q["flag"] for q in QUALITIES.values() if q["flag"]])),
            case_sensitive=False,
        ),
        help="Render quality at the follow resolution framerates, respectively: "
        + ", ".join(
            reversed(
                [
                    f"{q['pixel_width']}x{q['pixel_height']} {q['frame_rate']}FPS"
                    for q in QUALITIES.values()
                    if q["flag"]
                ]
            )
        ),
    ),
    option(
        "-r",
        "--resolution",
        callback=validate_resolution,
        default=None,
        help='Resolution in "W,H" for when 16:9 aspect ratio isn\'t possible.',
    ),
    option(
        "--fps",
        "--frame_rate",
        "frame_rate",
        type=float,
        default=None,
        help="Render at this frame rate.",
    ),
    option(
        "--renderer",
        type=Choice(
            [renderer_type.value for renderer_type in RendererType],
            case_sensitive=False,
        ),
        help="Select a renderer for your Scene.",
        default="cairo",
    ),
    option(
        "-g",
        "--save_pngs",
        is_flag=True,
        default=None,
        help="Save each frame as png (Deprecated).",
    ),
    option(
        "-i",
        "--save_as_gif",
        default=None,
        is_flag=True,
        help="Save as a gif (Deprecated).",
    ),
    option(
        "--save_sections",
        default=None,
        is_flag=True,
        help="Save section videos in addition to movie file.",
    ),
    option(
        "-t",
        "--transparent",
        is_flag=True,
        help="Render scenes with alpha channel.",
    ),
    option(
        "--use_projection_fill_shaders",
        is_flag=True,
        help="Use shaders for OpenGLVMobject fill which are compatible with transformation matrices.",
        default=None,
    ),
    option(
        "--use_projection_stroke_shaders",
        is_flag=True,
        help="Use shaders for OpenGLVMobject stroke which are compatible with transformation matrices.",
        default=None,
    ),
)
