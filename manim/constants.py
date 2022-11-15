"""
Constant definitions.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from cloup import Context
from PIL.Image import Resampling

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
]
# Messages

SCENE_NOT_FOUND_MESSAGE: str = """
   {} is not in the script
"""
CHOOSE_NUMBER_MESSAGE: str = """
Choose number corresponding to desired scene/arguments.
(Use comma separated list for multiple entries)
Choice(s): """
INVALID_NUMBER_MESSAGE: str = "Invalid scene numbers have been specified. Aborting."
NO_SCENE_MESSAGE: str = """
   There are no scenes inside that module
"""

# Pango stuff
NORMAL: str = "NORMAL"
ITALIC: str = "ITALIC"
OBLIQUE: str = "OBLIQUE"
BOLD: str = "BOLD"
# Only for Pango from below
THIN: str = "THIN"
ULTRALIGHT: str = "ULTRALIGHT"
LIGHT: str = "LIGHT"
SEMILIGHT: str = "SEMILIGHT"
BOOK: str = "BOOK"
MEDIUM: str = "MEDIUM"
SEMIBOLD: str = "SEMIBOLD"
ULTRABOLD: str = "ULTRABOLD"
HEAVY: str = "HEAVY"
ULTRAHEAVY: str = "ULTRAHEAVY"

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
ORIGIN: np.ndarray = np.array((0.0, 0.0, 0.0))
"""The center of the coordinate system."""

UP: np.ndarray = np.array((0.0, 1.0, 0.0))
"""One unit step in the positive Y direction."""

DOWN: np.ndarray = np.array((0.0, -1.0, 0.0))
"""One unit step in the negative Y direction."""

RIGHT: np.ndarray = np.array((1.0, 0.0, 0.0))
"""One unit step in the positive X direction."""

LEFT: np.ndarray = np.array((-1.0, 0.0, 0.0))
"""One unit step in the negative X direction."""

IN: np.ndarray = np.array((0.0, 0.0, -1.0))
"""One unit step in the negative Z direction."""

OUT: np.ndarray = np.array((0.0, 0.0, 1.0))
"""One unit step in the positive Z direction."""

# Geometry: axes
X_AXIS: np.ndarray = np.array((1.0, 0.0, 0.0))
Y_AXIS: np.ndarray = np.array((0.0, 1.0, 0.0))
Z_AXIS: np.ndarray = np.array((0.0, 0.0, 1.0))

# Geometry: useful abbreviations for diagonals
UL: np.ndarray = UP + LEFT
"""One step up plus one step left."""

UR: np.ndarray = UP + RIGHT
"""One step up plus one step right."""

DL: np.ndarray = DOWN + LEFT
"""One step down plus one step left."""

DR: np.ndarray = DOWN + RIGHT
"""One step down plus one step right."""

# Geometry
START_X: int = 30
START_Y: int = 20
DEFAULT_DOT_RADIUS: float = 0.08
DEFAULT_SMALL_DOT_RADIUS: float = 0.04
DEFAULT_DASH_LENGTH: float = 0.05
DEFAULT_ARROW_TIP_LENGTH: float = 0.35

# Default buffers (padding)
SMALL_BUFF: float = 0.1
MED_SMALL_BUFF: float = 0.25
MED_LARGE_BUFF: float = 0.5
LARGE_BUFF: float = 1
DEFAULT_MOBJECT_TO_EDGE_BUFFER: float = MED_LARGE_BUFF
DEFAULT_MOBJECT_TO_MOBJECT_BUFFER: float = MED_SMALL_BUFF

# Times in seconds
DEFAULT_POINTWISE_FUNCTION_RUN_TIME: float = 3.0
DEFAULT_WAIT_TIME: float = 1.0

# Misc
DEFAULT_POINT_DENSITY_2D: int = 25
DEFAULT_POINT_DENSITY_1D: int = 10
DEFAULT_STROKE_WIDTH: int = 4
DEFAULT_FONT_SIZE: float = 48

# Mathematical constants
PI: float = np.pi
"""The ratio of the circumference of a circle to its diameter."""

TAU: float = 2 * PI
"""The ratio of the circumference of a circle to its radius."""

DEGREES: float = TAU / 360
"""The exchange rate between radians and degrees."""

# Video qualities
QUALITIES: dict[str, dict[str, str | int | None]] = {
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

DEFAULT_QUALITY: str = "high_quality"

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
