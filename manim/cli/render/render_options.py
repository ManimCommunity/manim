from __future__ import annotations

import re

import click
from cloup import option, option_group

from manim.constants import QUALITIES, RendererType

from ... import logger


def validate_scene_range(ctx, param, value):
    try:
        start = int(value)
        return (start,)
    except Exception:
        pass

    if value:
        try:
            start, end = map(int, re.split(r"[;,\-]", value))
            return start, end
        except Exception:
            logger.error("Couldn't determine a range for -n option.")
            exit()


def validate_resolution(ctx, param, value):
    if value:
        try:
            start, end = map(int, re.split(r"[;,\-]", value))
            return (start, end)
        except Exception:
            logger.error("Resolution option is invalid.")
            exit()


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
        type=click.Choice(["png", "gif", "mp4", "webm", "mov"], case_sensitive=False),
        default=None,
    ),
    option("-s", "--save_last_frame", is_flag=True, default=None),
    option(
        "-q",
        "--quality",
        default=None,
        type=click.Choice(
            list(reversed([q["flag"] for q in QUALITIES.values() if q["flag"]])),  # type: ignore
            case_sensitive=False,
        ),
        help="Render quality at the follow resolution framerates, respectively: "
        + ", ".join(
            reversed(
                [
                    f'{q["pixel_width"]}x{q["pixel_height"]} {q["frame_rate"]}FPS'
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
        type=click.Choice(
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
        "-s",
        "--save_last_frame",
        default=None,
        is_flag=True,
        help="Save last frame as png (Deprecated).",
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
