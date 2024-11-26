"""Constant definitions."""

from __future__ import annotations

from enum import Enum
from typing import TypedDict

import numpy as np
from cloup import Context
from PIL.Image import Resampling

from manim.typing import Vector3D

__all__ = [
    "SCENE_NOT_FOUND_MESSAGE",
    "CHOOSE_NUMBER_MESSAGE",
    "INVALID_NUMBER_MESSAGE",
    "NO_SCENE_MESSAGE",
    "NORMAL",
    "ITALIC",
    "OBLIQUE",
    "BOLD",
    "THIN",
    "ULTRALIGHT",
    "LIGHT",
    "SEMILIGHT",
    "BOOK",
    "MEDIUM",
    "SEMIBOLD",
    "ULTRABOLD",
    "HEAVY",
    "ULTRAHEAVY",
    "RESAMPLING_ALGORITHMS",
    "ORIGIN",
    "UP",
    "DOWN",
    "RIGHT",
    "LEFT",
    "IN",
    "OUT",
    "X_AXIS",
    "Y_AXIS",
    "Z_AXIS",
    "UL",
    "UR",
    "DL",
    "DR",
    "START_X",
    "START_Y",
    "DEFAULT_DOT_RADIUS",
    "DEFAULT_SMALL_DOT_RADIUS",
    "DEFAULT_DASH_LENGTH",
    "DEFAULT_ARROW_TIP_LENGTH",
    "SMALL_BUFF",
    "MED_SMALL_BUFF",
    "MED_LARGE_BUFF",
    "LARGE_BUFF",
    "DEFAULT_MOBJECT_TO_EDGE_BUFFER",
    "DEFAULT_MOBJECT_TO_MOBJECT_BUFFER",
    "DEFAULT_POINTWISE_FUNCTION_RUN_TIME",
    "DEFAULT_WAIT_TIME",
    "DEFAULT_POINT_DENSITY_2D",
    "DEFAULT_POINT_DENSITY_1D",
    "DEFAULT_STROKE_WIDTH",
    "DEFAULT_FONT_SIZE",
    "SCALE_FACTOR_PER_FONT_POINT",
    "PI",
    "TAU",
    "DEGREES",
    "QUALITIES",
    "DEFAULT_QUALITY",
    "EPILOG",
    "CONTEXT_SETTINGS",
    "SHIFT_VALUE",
    "CTRL_VALUE",
    "RendererType",
    "LineJointType",
    "CapStyleType",
]
# Messages

SCENE_NOT_FOUND_MESSAGE = """
   {} is not in the script
"""
CHOOSE_NUMBER_MESSAGE = """
Choose number corresponding to desired scene/arguments.
(Use comma separated list for multiple entries)
Choice(s): """
INVALID_NUMBER_MESSAGE = "Invalid scene numbers have been specified. Aborting."
NO_SCENE_MESSAGE = """
   There are no scenes inside that module
"""

# Pango stuff
NORMAL = "NORMAL"
ITALIC = "ITALIC"
OBLIQUE = "OBLIQUE"
BOLD = "BOLD"
# Only for Pango from below
THIN = "THIN"
ULTRALIGHT = "ULTRALIGHT"
LIGHT = "LIGHT"
SEMILIGHT = "SEMILIGHT"
BOOK = "BOOK"
MEDIUM = "MEDIUM"
SEMIBOLD = "SEMIBOLD"
ULTRABOLD = "ULTRABOLD"
HEAVY = "HEAVY"
ULTRAHEAVY = "ULTRAHEAVY"

RESAMPLING_ALGORITHMS = {
    "nearest": Resampling.NEAREST,
    "none": Resampling.NEAREST,
    "lanczos": Resampling.LANCZOS,
    "antialias": Resampling.LANCZOS,
    "bilinear": Resampling.BILINEAR,
    "linear": Resampling.BILINEAR,
    "bicubic": Resampling.BICUBIC,
    "cubic": Resampling.BICUBIC,
    "box": Resampling.BOX,
    "hamming": Resampling.HAMMING,
}

# Geometry: directions
ORIGIN: Vector3D = np.array((0.0, 0.0, 0.0))
"""The center of the coordinate system."""

UP: Vector3D = np.array((0.0, 1.0, 0.0))
"""One unit step in the positive Y direction."""

DOWN: Vector3D = np.array((0.0, -1.0, 0.0))
"""One unit step in the negative Y direction."""

RIGHT: Vector3D = np.array((1.0, 0.0, 0.0))
"""One unit step in the positive X direction."""

LEFT: Vector3D = np.array((-1.0, 0.0, 0.0))
"""One unit step in the negative X direction."""

IN: Vector3D = np.array((0.0, 0.0, -1.0))
"""One unit step in the negative Z direction."""

OUT: Vector3D = np.array((0.0, 0.0, 1.0))
"""One unit step in the positive Z direction."""

# Geometry: axes
X_AXIS: Vector3D = np.array((1.0, 0.0, 0.0))
Y_AXIS: Vector3D = np.array((0.0, 1.0, 0.0))
Z_AXIS: Vector3D = np.array((0.0, 0.0, 1.0))

# Geometry: useful abbreviations for diagonals
UL: Vector3D = UP + LEFT
"""One step up plus one step left."""

UR: Vector3D = UP + RIGHT
"""One step up plus one step right."""

DL: Vector3D = DOWN + LEFT
"""One step down plus one step left."""

DR: Vector3D = DOWN + RIGHT
"""One step down plus one step right."""

# Geometry
START_X = 30
START_Y = 20
DEFAULT_DOT_RADIUS = 0.08
DEFAULT_SMALL_DOT_RADIUS = 0.04
DEFAULT_DASH_LENGTH = 0.05
DEFAULT_ARROW_TIP_LENGTH = 0.35

# Default buffers (padding)
SMALL_BUFF = 0.1
MED_SMALL_BUFF = 0.25
MED_LARGE_BUFF = 0.5
LARGE_BUFF = 1
DEFAULT_MOBJECT_TO_EDGE_BUFFER = MED_LARGE_BUFF
DEFAULT_MOBJECT_TO_MOBJECT_BUFFER = MED_SMALL_BUFF

# Times in seconds
DEFAULT_POINTWISE_FUNCTION_RUN_TIME = 3.0
DEFAULT_WAIT_TIME = 1.0

# Misc
DEFAULT_POINT_DENSITY_2D = 25
DEFAULT_POINT_DENSITY_1D = 10
DEFAULT_STROKE_WIDTH = 4
DEFAULT_FONT_SIZE = 48
SCALE_FACTOR_PER_FONT_POINT = 1 / 960

# Mathematical constants
PI = np.pi
"""The ratio of the circumference of a circle to its diameter."""

TAU = 2 * PI
"""The ratio of the circumference of a circle to its radius."""

DEGREES = TAU / 360
"""The exchange rate between radians and degrees."""


class QualityDict(TypedDict):
    flag: str | None
    pixel_height: int
    pixel_width: int
    frame_rate: int


# Video qualities
QUALITIES: dict[str, QualityDict] = {
    "fourk_quality": {
        "flag": "k",
        "pixel_height": 2160,
        "pixel_width": 3840,
        "frame_rate": 60,
    },
    "production_quality": {
        "flag": "p",
        "pixel_height": 1440,
        "pixel_width": 2560,
        "frame_rate": 60,
    },
    "high_quality": {
        "flag": "h",
        "pixel_height": 1080,
        "pixel_width": 1920,
        "frame_rate": 60,
    },
    "medium_quality": {
        "flag": "m",
        "pixel_height": 720,
        "pixel_width": 1280,
        "frame_rate": 30,
    },
    "low_quality": {
        "flag": "l",
        "pixel_height": 480,
        "pixel_width": 854,
        "frame_rate": 15,
    },
    "example_quality": {
        "flag": None,
        "pixel_height": 480,
        "pixel_width": 854,
        "frame_rate": 30,
    },
}

DEFAULT_QUALITY = "high_quality"

EPILOG = "Made with <3 by Manim Community developers."
SHIFT_VALUE = 65505
CTRL_VALUE = 65507

CONTEXT_SETTINGS = Context.settings(
    align_option_groups=True,
    align_sections=True,
    show_constraints=True,
)


class RendererType(Enum):
    """An enumeration of all renderer types that can be assigned to
    the ``config.renderer`` attribute.

    Manim's configuration allows assigning string values to the renderer
    setting, the values are then replaced by the corresponding enum object.
    In other words, you can run::

        config.renderer = "opengl"

    and checking the renderer afterwards reveals that the attribute has
    assumed the value::

        <RendererType.OPENGL: 'opengl'>
    """

    CAIRO = "cairo"  #: A renderer based on the cairo backend.
    OPENGL = "opengl"  #: An OpenGL-based renderer.


class LineJointType(Enum):
    """Collection of available line joint types.

    See the example below for a visual illustration of the different
    joint types.

    Examples
    --------

    .. manim:: LineJointVariants
        :save_last_frame:

        class LineJointVariants(Scene):
            def construct(self):
                mob = VMobject(stroke_width=20, color=GREEN).set_points_as_corners([
                    np.array([-2, 0, 0]),
                    np.array([0, 0, 0]),
                    np.array([-2, 1, 0]),
                ])
                lines = VGroup(*[mob.copy() for _ in range(len(LineJointType))])
                for line, joint_type in zip(lines, LineJointType):
                    line.joint_type = joint_type

                lines.arrange(RIGHT, buff=1)
                self.add(lines)
                for line in lines:
                    label = Text(line.joint_type.name).next_to(line, DOWN)
                    self.add(label)
    """

    AUTO = 0
    ROUND = 1
    BEVEL = 2
    MITER = 3


class CapStyleType(Enum):
    """Collection of available cap styles.

    See the example below for a visual illustration of the different
    cap styles.

    Examples
    --------

    .. manim:: CapStyleVariants
        :save_last_frame:

        class CapStyleVariants(Scene):
            def construct(self):
                arcs = VGroup(*[
                    Arc(
                        radius=1,
                        start_angle=0,
                        angle=TAU / 4,
                        stroke_width=20,
                        color=GREEN,
                        cap_style=cap_style,
                    )
                    for cap_style in CapStyleType
                ])
                arcs.arrange(RIGHT, buff=1)
                self.add(arcs)
                for arc in arcs:
                    label = Text(arc.cap_style.name, font_size=24).next_to(arc, DOWN)
                    self.add(label)
    """

    AUTO = 0
    ROUND = 1
    BUTT = 2
    SQUARE = 3
