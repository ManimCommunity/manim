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
    "Colors",
]

import random
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

    .. manim:: ColorExample
        :save_last_frame:

        from manim.utils.color import Colors
        class ColorExample(Scene):
            def construct(self):
                cols = Colors._member_names_
                s = VGroup(*[Line(DOWN, UP, stroke_width=15).set_color(Colors[cols[i]].value) for i in range(0, len(cols))])
                s.arrange_submobjects(buff=0.2)
                self.add(s)

    The preferred way of using these colors is

    .. code-block:: pycon

        >>> import manim.utils.color as C
        >>> C.WHITE
        '#FFFFFF'

    Note this way uses the name of the colors in UPPERCASE.

    Alternatively, you can also import this Enum directly and use its members
    directly, through the use of :code:`color.value`.  Note this way uses the
    name of the colors in lowercase.

    .. code-block:: pycon

        >>> from manim.utils.color import Colors
        >>> Colors.white.value
        '#FFFFFF'

    """

    white = "#FFFFFF"
    gray_a = "#DDDDDD"
    gray_b = "#BBBBBB"
    gray_c = "#888888"
    gray_d = "#444444"
    gray_e = "#222222"
    black = "#000000"
    lighter_gray = gray_a
    light_gray = gray_b
    gray = gray_c
    dark_gray = gray_d
    darker_gray = gray_e

    blue_a = "#C7E9F1"
    blue_b = "#9CDCEB"
    blue_c = "#58C4DD"
    blue_d = "#29ABCA"
    blue_e = "#1C758A"
    dark_blue = "#236B8E"
    pure_blue = "#0000FF"
    blue = blue_c

    teal_a = "#ACEAD7"
    teal_b = "#76DDC0"
    teal_c = "#5CD0B3"
    teal_d = "#55C1A7"
    teal_e = "#49A88F"
    teal = teal_c

    green_a = "#C9E2AE"
    green_b = "#A6CF8C"
    green_c = "#83C167"
    green_d = "#77B05D"
    green_e = "#699C52"
    pure_green = "#00FF00"
    green = green_c

    yellow_a = "#FFF1B6"
    yellow_b = "#FFEA94"
    yellow_c = "#FFFF00"
    yellow_e = "#E8C11C"
    yellow_d = "#F4D345"
    yellow = yellow_c

    gold_a = "#F7C797"
    gold_b = "#F9B775"
    gold_c = "#F0AC5F"
    gold_d = "#E1A158"
    gold_e = "#C78D46"
    gold = gold_c

    red_a = "#F7A1A3"
    red_b = "#FF8080"
    red_c = "#FC6255"
    red_d = "#E65A4C"
    red_e = "#CF5044"
    pure_red = "#FF0000"
    red = red_c

    maroon_a = "#ECABC1"
    maroon_b = "#EC92AB"
    maroon_c = "#C55F73"
    maroon_d = "#A24D61"
    maroon_e = "#94424F"
    maroon = maroon_c

    purple_a = "#CAA3E8"
    purple_b = "#B189C6"
    purple_c = "#9A72AC"
    purple_d = "#715582"
    purple_e = "#644172"
    purple = purple_c

    pink = "#D147BD"
    light_pink = "#DC75CD"

    orange = "#FF862F"
    light_brown = "#CD853F"
    dark_brown = "#8B4513"
    gray_brown = "#736357"


# Create constants from Colors enum
constants_names = []
for name, value in Colors.__members__.items():
    name = name.upper()
    constants_names.append(name)
    locals()[name] = value
    if "GRAY" in name:
        name = name.replace("GRAY", "GREY")
        locals()[name] = value
        constants_names.append(name)

# Add constants to module exports
__all__ += constants_names


def color_to_rgb(color):
    if isinstance(color, str):
        return hex_to_rgb(color)
    elif isinstance(color, Color):
        return np.array(color.get_rgb())
    else:
        raise ValueError("Invalid color type")


def color_to_rgba(color, alpha=1):
    return np.array([*color_to_rgb(color), alpha])


def rgb_to_color(rgb):
    try:
        return Color(rgb=rgb)
    except Exception:
        return Color(WHITE)


def rgba_to_color(rgba):
    return rgb_to_color(rgba[:3])


def rgb_to_hex(rgb):
    return "#" + "".join("%02x" % int(255 * x) for x in rgb)


def hex_to_rgb(hex_code):
    hex_part = hex_code[1:]
    if len(hex_part) == 3:
        hex_part = "".join([2 * c for c in hex_part])
    return np.array([int(hex_part[i : i + 2], 16) / 255 for i in range(0, 6, 2)])


def invert_color(color):
    return rgb_to_color(1.0 - color_to_rgb(color))


def color_to_int_rgb(color):
    return (255 * color_to_rgb(color)).astype("uint8")


def color_to_int_rgba(color, opacity=1.0):
    alpha = int(255 * opacity)
    return np.append(color_to_int_rgb(color), alpha)


def color_gradient(reference_colors, length_of_output):
    if length_of_output == 0:
        return reference_colors[0]
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


def interpolate_color(color1, color2, alpha):
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
