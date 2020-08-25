"""
Constant definitions.
"""

from enum import Enum
import numpy as np


# Messages
NOT_SETTING_FONT_MSG = """
You haven't set font.
If you are not using English, this may cause text rendering problem.
You set font like:
text = Text('your text', font='your font')
or:
class MyText(Text):
    CONFIG = {
        'font': 'My Font'
    }
"""
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

# Cairo stuff
NORMAL = "NORMAL"
ITALIC = "ITALIC"
OBLIQUE = "OBLIQUE"
BOLD = "BOLD"

# Geometry: directions
ORIGIN = np.array((0.0, 0.0, 0.0))
"""The center of the coordinate system."""

UP = np.array((0.0, 1.0, 0.0))
"""One unit step in the positive Y direction."""

DOWN = np.array((0.0, -1.0, 0.0))
"""One unit step in the negative Y direction."""

RIGHT = np.array((1.0, 0.0, 0.0))
"""One unit step in the positive X direction."""

LEFT = np.array((-1.0, 0.0, 0.0))
"""One unit step in the negative X direction."""

IN = np.array((0.0, 0.0, -1.0))
"""One unit step in the negative Z direction."""

OUT = np.array((0.0, 0.0, 1.0))
"""One unit step in the positive Z direction."""

# Geometry: axes
X_AXIS = np.array((1.0, 0.0, 0.0))
Y_AXIS = np.array((0.0, 1.0, 0.0))
Z_AXIS = np.array((0.0, 0.0, 1.0))

# Geometry: useful abbreviations for diagonals
UL = UP + LEFT
"""One step up plus one step left."""

UR = UP + RIGHT
"""One step up plus one step right."""

DL = DOWN + LEFT
"""One step down plus one step left."""

DR = DOWN + RIGHT
"""One step down plus one step right."""

# Geometry
START_X = 30
START_Y = 20

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
DEFAULT_POINT_DENSITY_1D = 250
DEFAULT_STROKE_WIDTH = 4

# Mathematical constants
PI = np.pi
"""The ratio of the circumference of a circle to its diameter."""

TAU = 2 * PI
"""The ratio of the circumference of a circle to its radius."""

DEGREES = TAU / 360
"""The exchange rate between radians and degrees."""

# ffmpeg stuff
FFMPEG_BIN = "ffmpeg"
FFMPEG_VERBOSITY_MAP = {
    "DEBUG": "error",
    "INFO": "error",
    "WARNING": "error",
    "ERROR": "error",
    "CRITICAL": "fatal",
}
VERBOSITY_CHOICES = FFMPEG_VERBOSITY_MAP.keys()

# gif stuff
GIF_FILE_EXTENSION = ".gif"


# Colors
class Colors(Enum):
    """A list of pre-defined colors."""
    dark_blue = "#236B8E"
    dark_brown = "#8B4513"
    light_brown = "#CD853F"
    blue_e = "#1C758A"
    blue_d = "#29ABCA"
    blue_c = "#58C4DD"
    blue = "#58C4DD"
    blue_b = "#9CDCEB"
    blue_a = "#C7E9F1"
    teal_e = "#49A88F"
    teal_d = "#55C1A7"
    teal_c = "#5CD0B3"
    teal = "#5CD0B3"
    teal_b = "#76DDC0"
    teal_a = "#ACEAD7"
    green_e = "#699C52"
    green_d = "#77B05D"
    green_c = "#83C167"
    green = "#83C167"
    green_b = "#A6CF8C"
    green_a = "#C9E2AE"
    yellow_e = "#E8C11C"
    yellow_d = "#F4D345"
    yellow_c = "#FFFF00"
    yellow = "#FFFF00"
    yellow_b = "#FFEA94"
    yellow_a = "#FFF1B6"
    gold_e = "#C78D46"
    gold_d = "#E1A158"
    gold_c = "#F0AC5F"
    gold = "#F0AC5F"
    gold_b = "#F9B775"
    gold_a = "#F7C797"
    red_e = "#CF5044"
    red_d = "#E65A4C"
    red_c = "#FC6255"
    red = "#FC6255"
    red_b = "#FF8080"
    red_a = "#F7A1A3"
    maroon_e = "#94424F"
    maroon_d = "#A24D61"
    maroon_c = "#C55F73"
    maroon = "#C55F73"
    maroon_b = "#EC92AB"
    maroon_a = "#ECABC1"
    purple_e = "#644172"
    purple_d = "#715582"
    purple_c = "#9A72AC"
    purple = "#9A72AC"
    purple_b = "#B189C6"
    purple_a = "#CAA3E8"
    white = "#FFFFFF"
    black = "#000000"
    light_gray = "#BBBBBB"
    light_grey = "#BBBBBB"
    gray = "#888888"
    grey = "#888888"
    dark_grey = "#444444"
    dark_gray = "#444444"
    darker_grey = "#222222"
    darker_gray = "#222222"
    grey_brown = "#736357"
    pink = "#D147BD"
    light_pink = "#DC75CD"
    green_screen = "#00FF00"
    orange = "#FF862F"
