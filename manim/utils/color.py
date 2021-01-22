"""Utility functions for conversion between color models."""

__all__ = [
    "color_to_rgb",
    "color_to_rgba",
    "rgb_to_color",
    "rgba_to_color",
    "rgb_to_hex",
    "hex_to_rgb",
    "invert_color",
    "color_to_int_rgb",
    "color_to_int_rgba",
    "color_gradient",
    "interpolate_color",
    "average_color",
    "random_bright_color",
    "random_color",
    "get_shaded_rgb",
    "DARK_BLUE",
    "DARK_BROWN",
    "LIGHT_BROWN",
    "BLUE_E",
    "BLUE_D",
    "BLUE_C",
    "BLUE",
    "BLUE_B",
    "BLUE_A",
    "TEAL_E",
    "TEAL_D",
    "TEAL_C",
    "TEAL",
    "TEAL_B",
    "TEAL_A",
    "GREEN_E",
    "GREEN_D",
    "GREEN_C",
    "GREEN",
    "GREEN_B",
    "GREEN_A",
    "YELLOW_E",
    "YELLOW_D",
    "YELLOW_C",
    "YELLOW",
    "YELLOW_B",
    "YELLOW_A",
    "GOLD_E",
    "GOLD_D",
    "GOLD_C",
    "GOLD",
    "GOLD_B",
    "GOLD_A",
    "RED_E",
    "RED_D",
    "RED_C",
    "RED",
    "RED_B",
    "RED_A",
    "MAROON_E",
    "MAROON_D",
    "MAROON_C",
    "MAROON",
    "MAROON_B",
    "MAROON_A",
    "PURPLE_E",
    "PURPLE_D",
    "PURPLE_C",
    "PURPLE",
    "PURPLE_B",
    "PURPLE_A",
    "WHITE",
    "BLACK",
    "LIGHT_GRAY",
    "LIGHT_GREY",
    "GRAY",
    "GREY",
    "DARK_GREY",
    "DARK_GRAY",
    "DARKER_GREY",
    "DARKER_GRAY",
    "GREY_BROWN",
    "PINK",
    "LIGHT_PINK",
    "GREEN_SCREEN",
    "ORANGE",
]

import random
import re
import typing
from enum import Enum

import numpy as np
from colour import Color

from ..utils.bezier import interpolate
from ..utils.simple_functions import clip_in_place
from ..utils.space_ops import normalize


class Colors(Enum):
    """A list of pre-defined colors.

    Examples
    --------
    The preferred way of using these colors is

    .. code-block:: python

        >>> import manim.utils.color as C
        >>> C.WHITE
        '#FFFFFF'

    Note this way uses the name of the colors in UPPERCASE.

    Alternatively, you can also import this Enum directly and use its members
    directly, through the use of :code:`color.value`.  Note this way uses the
    name of the colors in lowercase.

    .. code-block:: python

        >>> from manim.utils.color import Colors
        >>> Colors.white.value
        '#FFFFFF'

    """

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


DARK_BLUE = Colors.dark_blue.value
DARK_BROWN = Colors.dark_brown.value
LIGHT_BROWN = Colors.dark_brown.value
BLUE_E = Colors.blue_e.value
BLUE_D = Colors.blue_d.value
BLUE_C = Colors.blue_c.value
BLUE = Colors.blue.value
BLUE_B = Colors.blue_b.value
BLUE_A = Colors.blue_a.value
TEAL_E = Colors.teal_e.value
TEAL_D = Colors.teal_d.value
TEAL_C = Colors.teal_c.value
TEAL = Colors.teal.value
TEAL_B = Colors.teal_b.value
TEAL_A = Colors.teal_a.value
GREEN_E = Colors.green_e.value
GREEN_D = Colors.green_d.value
GREEN_C = Colors.green_c.value
GREEN = Colors.green.value
GREEN_B = Colors.green_b.value
GREEN_A = Colors.green_a.value
YELLOW_E = Colors.yellow_e.value
YELLOW_D = Colors.yellow_d.value
YELLOW_C = Colors.yellow_c.value
YELLOW = Colors.yellow.value
YELLOW_B = Colors.yellow_b.value
YELLOW_A = Colors.yellow_a.value
GOLD_E = Colors.gold_e.value
GOLD_D = Colors.gold_d.value
GOLD_C = Colors.gold_c.value
GOLD = Colors.gold.value
GOLD_B = Colors.gold_b.value
GOLD_A = Colors.gold_a.value
RED_E = Colors.red_e.value
RED_D = Colors.red_d.value
RED_C = Colors.red_c.value
RED = Colors.red.value
RED_B = Colors.red_b.value
RED_A = Colors.red_a.value
MAROON_E = Colors.maroon_e.value
MAROON_D = Colors.maroon_d.value
MAROON_C = Colors.maroon_c.value
MAROON = Colors.maroon.value
MAROON_B = Colors.maroon_b.value
MAROON_A = Colors.maroon_a.value
PURPLE_E = Colors.purple_e.value
PURPLE_D = Colors.purple_d.value
PURPLE_C = Colors.purple_c.value
PURPLE = Colors.purple.value
PURPLE_B = Colors.purple_b.value
PURPLE_A = Colors.purple_a.value
WHITE = Colors.white.value
BLACK = Colors.black.value
LIGHT_GRAY = Colors.light_gray.value
LIGHT_GREY = Colors.light_grey.value
GRAY = Colors.gray.value
GREY = Colors.grey.value
DARK_GREY = Colors.dark_grey.value
DARK_GRAY = Colors.dark_gray.value
DARKER_GREY = Colors.darker_gray.value
DARKER_GRAY = Colors.darker_gray.value
GREY_BROWN = Colors.grey_brown.value
PINK = Colors.pink.value
LIGHT_PINK = Colors.light_pink.value
GREEN_SCREEN = Colors.green_screen.value
ORANGE = Colors.orange.value

Colour = typing.Union[str, Color]
# numpy doesn't support generics so this is the best I can do
RGB = np.array
RGBA = np.array
INT_RGB = np.array
INT_RGBA = np.array


def color_to_rgb(color: Colour) -> RGB:
    """
    Convert `color` to RGB

    `color` must be either a `colour.Color` or a hexadecimal colour string.

    :param color: The colour to convert
    :type color: colour.Color | str
    :raises ValueError: `color` is not a valid colour
    :return: RGB colour
    :rtype: numpy.ndarray[numpy.float64]
    """
    if isinstance(color, str):
        try:
            return hex_to_rgb(color)
        except ValueError as exc:
            # the default message is not meaningful
            # so here's a custom one
            raise ValueError("Invalid hex colour") from exc
    elif isinstance(color, Color):
        return np.array(color.get_rgb())
    else:
        raise ValueError("Invalid color type")


def color_to_rgba(color: Colour, alpha: float = 1) -> RGBA:
    """
    Convert `color` to RGBA using the given `alpha`

    `color` must be either a `colour.Color` or a hexadecimal colour string.

    :param color: The colour to convert
    :type color: colour.Color | str
    :param alpha: The alpha value, in the range [0, 1]
    :raises ValueError: `color` is not a valid colour
    :return: RGB colour
    :rtype: numpy.ndarray[numpy.float64]
    """
    return np.array([*color_to_rgb(color), alpha])


def rgb_to_color(rgb: RGB) -> Color:
    """
    Convert `rgb` to a `colour.Color`

    :param rgb: 3-array of floats in the range [0, 1]
    :type rgb: numpy.ndarray[numpy.float64]
    :raises ValueError: `rgb` is not a valid RGB colour
    :return: The converted colour
    :rtype: colour.Color
    """
    try:
        return Color(rgb=rgb)
    except ValueError as exc:
        raise ValueError("Not a valid RGB colour") from exc


def rgba_to_color(rgba: RGBA) -> Color:
    """
    Convert `rgba` to a `colour.Color`

    The alpha channel is stripped (and is therefore lost).

    :param rgba: 4-array of floats in the range [0, 1]
    :type rgba: numpy.ndarray[numpy.float64]
    :raises ValueError: `rgba` is not a valid RGB colour
    :return: The converted colour
    :rtype: colour.Color
    """
    return rgb_to_color(rgba[:3])


def rgb_to_hex(rgb: typing.Union[RGB, RGBA]) -> str:
    """
    Convert `rgb` to a (web-safe) hexadecimal colour string

    `rgb` may also be an RGBA value, but for backwards compatibility
    I have left its name as `rgb`

    :param rgb: 3- or 4- array of floats in the range [0, 1]
    :type rgb: numpy.ndarray[numpy.float64]
    :raises ValueError: `rgb` is not a valid RGB/RGBA colour
    :return: Hexadecimal colour string
    :rtype: str
    """
    if not (3 <= len(rgb) <= 4):
        raise ValueError("Not an RGB/RGBA colour")
    return "#" + "".join("%02x" % int(255 * x) for x in rgb)


def hex_to_rgb(hex_code: str) -> RGB:
    """
    Convert `hex_code` to an RGB array

    :param hex_code: Hexadecimal colour string
    :type hex_code: str
    :raises ValueError: `hex_code` is not a valid hexadecimal colour
    :return: RGB array
    :rtype: numpy.ndarray[numpy.float64]
    """
    if not re.fullmatch(r"#([0-9a-f]{3}|[0-9a-f]{6})", hex_code, flags=re.I):
        raise ValueError("Not a valid hexadecimal colour")
    hex_part = hex_code[1:]
    if len(hex_part) == 3:
        hex_part = "".join([2 * c for c in hex_part])
    return np.array([int(hex_part[i : i + 2], 16) / 255 for i in range(0, 6, 2)])


def invert_color(color: Colour) -> Color:
    """
    Invert `color`

    `color` must be a `colour.Color` or a hexadecimal colour string

    :param color: The colour to invert
    :type color: colour.Color | str
    :return: The inverted colour
    :rtype: colour.Color
    """
    return rgb_to_color(1.0 - color_to_rgb(color))


def color_to_int_rgb(color: Colour) -> INT_RGB:
    """
    Convert `color` to an integer RGB array

    `color` must be a `colour.Color` or a hexadecimal colour string

    :param color: The colour to convert
    :type color: colour.Color | str
    :return: Integer RGB array
    :rtype: numpy.ndarray[numpy.uint8]
    """
    return (255 * color_to_rgb(color)).astype("uint8")


def color_to_int_rgba(color: Colour, opacity: float = 1.0) -> INT_RGBA:
    """
    Convert `color` to an integer RGBA array with the given `opacity`

    `color` must be a `colour.Color` or a hexadecimal colour string

    :param color: The colour to convert
    :type color: colour.Color | str
    :param opacity: The opacity of the converted colour, in the range [0, 1]
    :type opacity: float
    :return: Integer RGBA array
    :rtype: numpy.ndarray[numpy.uint8]
    """
    alpha = int(255 * opacity)
    return np.append(color_to_int_rgb(color), alpha)


def color_gradient(
    reference_colors: typing.List[Colour], length_of_output: int
) -> typing.List[Color]:
    """
    Compute and return a gradient based on `reference_colors`

    The reference colours will be turned into a `length_of_output`-long
    list of colours

    :param reference_colors: The colours with which to generate the gradient
    :type reference_colors: list[colour.Color | str]
    :param length_of_output: How many colours to return
    :type length_of_output: int
    :return: A list of colours in the gradient
    :rtype: list[colour.Color]
    """
    if length_of_output < 1:
        # I believe this to make more sense than just returning the first.
        # Should be backwards-compatible as I don't see why anyone would be
        # doing this in the first place
        return []
    rgbs = list(map(color_to_rgb, reference_colors))
    alphas = np.linspace(0, (len(rgbs) - 1), length_of_output)
    floors = alphas.astype("int")
    alphas_mod1 = alphas % 1
    # End edge case
    alphas_mod1[-1] = 1
    floors[-1] = len(rgbs) - 2
    return [
        rgb_to_color(interpolate(rgbs[i], rgbs[i + 1], alpha))
        for i, alpha in zip(floors, alphas_mod1)
    ]


def interpolate_color(color1: Colour, color2: Colour, alpha: float) -> Color:
    """
    Blend two colours using RGB interpolation

    Return a colour whose value is (alpha)-way between `color1` and `color2`

    :param color1: The base colour
    :type color1: colour.Color | str
    :param color2: The target colour
    :type color2: colour.Color | str
    :param alpha: How much of `color2` to blend into `color1`, in the range [0, 1]
    :type alpha: float
    :return: The blended colour
    :rtype: colour.Color
    """
    rgb = interpolate(color_to_rgb(color1), color_to_rgb(color2), alpha)
    return rgb_to_color(rgb)


def average_color(*colors):
    rgbs = np.array(list(map(color_to_rgb, colors)))
    mean_rgb = np.apply_along_axis(np.mean, 0, rgbs)
    return rgb_to_color(mean_rgb)


def random_bright_color():
    color = random_color()
    curr_rgb = color_to_rgb(color)
    new_rgb = interpolate(curr_rgb, np.ones(len(curr_rgb)), 0.5)
    return Color(rgb=new_rgb)


def random_color():
    return random.choice([c.value for c in list(Colors)])


def get_shaded_rgb(rgb, point, unit_normal_vect, light_source):
    to_sun = normalize(light_source - point)
    factor = 0.5 * np.dot(unit_normal_vect, to_sun) ** 3
    if factor < 0:
        factor *= 0.5
    result = rgb + factor
    clip_in_place(rgb + factor, 0, 1)
    return result
