from __future__ import annotations

import logging
import re
import sys
from typing import TYPE_CHECKING

from click import BadParameter
from cloup import Choice, IntRange, option, option_group

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


def validate_encoder_options(
    ctx: Context,
    param: Option,
    value: tuple[str, ...],
) -> dict[str, str] | None:
    """Parse repeatable ``KEY=VALUE`` encoder options."""
    if not value:
        return None

    options: dict[str, str] = {}
    for entry in value:
        key, separator, option_value = entry.partition("=")
        key = key.strip()
        if not separator or not key or not option_value:
            raise BadParameter(
                "encoder options must use KEY=VALUE with nonempty key and value",
                ctx=ctx,
                param=param,
            )
        if key in options:
            raise BadParameter(
                f"encoder option {key!r} was supplied more than once",
                ctx=ctx,
                param=param,
            )
        options[key] = option_value
    return options


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
        type=Choice(
            [
                "auto",
                "none",
                "png",
                "png-sequence",
                "gif",
                "mp4",
                "webm",
                "mov",
            ],
            case_sensitive=False,
        ),
        default=None,
        help="Primary output format. PNG renders only the last frame; "
        "png-sequence writes every rendered frame.",
    ),
    option(
        "-s",
        "--save_last_frame",
        default=None,
        is_flag=True,
        help="Fast-forward animations and save the last frame as PNG "
        "(equivalent to --format=png).",
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
        "--video-codec",
        "video_codec",
        default=None,
        help="Video encoder used for cached segments (default: auto).",
    ),
    option(
        "--pixel-format",
        "pixel_format",
        default=None,
        help="Pixel format used for cached segments (default: auto).",
    ),
    option(
        "--encoder-option",
        "video_encoder_options",
        multiple=True,
        default=None,
        callback=validate_encoder_options,
        metavar="KEY=VALUE",
        help="Set a codec option; repeat to set multiple options.",
    ),
    option(
        "--max-inflight-encoders",
        type=IntRange(min=1),
        default=None,
        help="Maximum number of partial movie files being encoded concurrently "
        "while the scene continues rendering. 1 (the default) encodes each "
        "animation's file before the next animation starts; values > 1 overlap "
        "encoding with rendering (4 is a good value on typical hardware).",
    ),
    option(
        "--encoder-queue-size",
        type=IntRange(min=1),
        default=None,
        help="Maximum number of pending frame buffers held by each encoder when "
        "parallel encoding is enabled. Ignored when --max-inflight-encoders is "
        "1 (the default). Defaults to 8.",
    ),
    option(
        "--renderer",
        type=Choice(
            [renderer_type.value for renderer_type in RendererType],
            case_sensitive=False,
        ),
        help="Select a renderer for your Scene.",
        default=None,
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
        default=None,
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
