"""Colors and utility functions for conversion between different color models."""

from __future__ import annotations

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
]

import random
from enum import Enum
from typing import Iterable

import numpy as np
from colour import Color

from ..utils.bezier import interpolate
from ..utils.space_ops import normalize


class Colors(Enum):
    """A list of pre-defined colors.

    Examples
    --------

    .. manim:: ColorsOverview
        :save_last_frame:
        :hide_source:

        from manim.utils.color import Colors
        class ColorsOverview(Scene):
            def construct(self):
                def color_group(color):
                    group = VGroup(
                        *[
                            Line(ORIGIN, RIGHT * 1.5, stroke_width=35, color=Colors[name].value)
                            for name in subnames(color)
                        ]
                    ).arrange_submobjects(buff=0.4, direction=DOWN)

                    name = Text(color).scale(0.6).next_to(group, UP, buff=0.3)
                    if any(decender in color for decender in "gjpqy"):
                        name.shift(DOWN * 0.08)
                    group.add(name)
                    return group

                def subnames(name):
                    return [name + "_" + char for char in "abcde"]

                color_groups = VGroup(
                    *[
                        color_group(color)
                        for color in [
                            "blue",
                            "teal",
                            "green",
                            "yellow",
                            "gold",
                            "red",
                            "maroon",
                            "purple",
                        ]
                    ]
                ).arrange_submobjects(buff=0.2, aligned_edge=DOWN)

                for line, char in zip(color_groups[0], "abcde"):
                    color_groups.add(Text(char).scale(0.6).next_to(line, LEFT, buff=0.2))

                def named_lines_group(length, colors, names, text_colors, align_to_block):
                    lines = VGroup(
                        *[
                            Line(
                                ORIGIN,
                                RIGHT * length,
                                stroke_width=55,
                                color=Colors[color].value,
                            )
                            for color in colors
                        ]
                    ).arrange_submobjects(buff=0.6, direction=DOWN)

                    for line, name, color in zip(lines, names, text_colors):
                        line.add(Text(name, color=color).scale(0.6).move_to(line))
                    lines.next_to(color_groups, DOWN, buff=0.5).align_to(
                        color_groups[align_to_block], LEFT
                    )
                    return lines

                other_colors = (
                    "pink",
                    "light_pink",
                    "orange",
                    "light_brown",
                    "dark_brown",
                    "gray_brown",
                )

                other_lines = named_lines_group(
                    3.2,
                    other_colors,
                    other_colors,
                    [BLACK] * 4 + [WHITE] * 2,
                    0,
                )

                gray_lines = named_lines_group(
                    6.6,
                    ["white"] + subnames("gray") + ["black"],
                    [
                        "white",
                        "lighter_gray / gray_a",
                        "light_gray / gray_b",
                        "gray / gray_c",
                        "dark_gray / gray_d",
                        "darker_gray / gray_e",
                        "black",
                    ],
                    [BLACK] * 3 + [WHITE] * 4,
                    2,
                )

                pure_colors = (
                    "pure_red",
                    "pure_green",
                    "pure_blue",
                )

                pure_lines = named_lines_group(
                    3.2,
                    pure_colors,
                    pure_colors,
                    [BLACK, BLACK, WHITE],
                    6,
                )

                self.add(color_groups, other_lines, gray_lines, pure_lines)

                VGroup(*self.mobjects).move_to(ORIGIN)


    The preferred way of using these colors is by importing their constants from manim:

    .. code-block:: pycon

        >>> from manim import RED, GREEN, BLUE
        >>> RED
        '#FC6255'

    Note this way uses the name of the colors in UPPERCASE.

    Alternatively, you can also import this Enum directly and use its members
    directly, through the use of :code:`color.value`.  Note this way uses the
    name of the colors in lowercase.

    .. code-block:: pycon

        >>> from manim.utils.color import Colors
        >>> Colors.red.value
        '#FC6255'

    .. note::

        The colors of type "C" have an alias equal to the colorname without a letter,
        e.g. GREEN = GREEN_C

    """

    white: str = "#FFFFFF"
    gray_a: str = "#DDDDDD"
    gray_b: str = "#BBBBBB"
    gray_c: str = "#888888"
    gray_d: str = "#444444"
    gray_e: str = "#222222"
    black: str = "#000000"
    lighter_gray: str = gray_a
    light_gray: str = gray_b
    gray: str = gray_c
    dark_gray: str = gray_d
    darker_gray: str = gray_e

    blue_a: str = "#C7E9F1"
    blue_b: str = "#9CDCEB"
    blue_c: str = "#58C4DD"
    blue_d: str = "#29ABCA"
    blue_e: str = "#236B8E"
    pure_blue: str = "#0000FF"
    blue: str = blue_c
    dark_blue: str = blue_e

    teal_a: str = "#ACEAD7"
    teal_b: str = "#76DDC0"
    teal_c: str = "#5CD0B3"
    teal_d: str = "#55C1A7"
    teal_e: str = "#49A88F"
    teal: str = teal_c

    green_a: str = "#C9E2AE"
    green_b: str = "#A6CF8C"
    green_c: str = "#83C167"
    green_d: str = "#77B05D"
    green_e: str = "#699C52"
    pure_green: str = "#00FF00"
    green: str = green_c

    yellow_a: str = "#FFF1B6"
    yellow_b: str = "#FFEA94"
    yellow_c: str = "#FFFF00"
    yellow_d: str = "#F4D345"
    yellow_e: str = "#E8C11C"
    yellow: str = yellow_c

    gold_a: str = "#F7C797"
    gold_b: str = "#F9B775"
    gold_c: str = "#F0AC5F"
    gold_d: str = "#E1A158"
    gold_e: str = "#C78D46"
    gold: str = gold_c

    red_a: str = "#F7A1A3"
    red_b: str = "#FF8080"
    red_c: str = "#FC6255"
    red_d: str = "#E65A4C"
    red_e: str = "#CF5044"
    pure_red: str = "#FF0000"
    red: str = red_c

    maroon_a: str = "#ECABC1"
    maroon_b: str = "#EC92AB"
    maroon_c: str = "#C55F73"
    maroon_d: str = "#A24D61"
    maroon_e: str = "#94424F"
    maroon: str = maroon_c

    purple_a: str = "#CAA3E8"
    purple_b: str = "#B189C6"
    purple_c: str = "#9A72AC"
    purple_d: str = "#715582"
    purple_e: str = "#644172"
    purple: str = purple_c

    pink: str = "#D147BD"
    light_pink: str = "#DC75CD"

    orange: str = "#FF862F"
    light_brown: str = "#CD853F"
    dark_brown: str = "#8B4513"
    gray_brown: str = "#736357"


def print_constant_definitions():
    """
    A simple function used to generate the constant values below. To run it
    paste this function and the Colors class into a file and run them.
    """
    constants_names: list[str] = []
    for name in Colors.__members__.keys():
        name_upper = name.upper()

        constants_names.append(name_upper)
        print(f"{name_upper} = Colors.{name}")

        if "GRAY" in name_upper:
            name_upper = name_upper.replace("GRAY", "GREY")

            constants_names.append(name_upper)
            print(f"{name_upper} = Colors.{name}")

    constants_names_repr = '[\n    "' + '",\n    "'.join(constants_names) + '",\n]'

    print(f"\n__all__ += {constants_names_repr}")


WHITE: str = "#FFFFFF"
GRAY_A: str = "#DDDDDD"
GREY_A: str = "#DDDDDD"
GRAY_B: str = "#BBBBBB"
GREY_B: str = "#BBBBBB"
GRAY_C: str = "#888888"
GREY_C: str = "#888888"
GRAY_D: str = "#444444"
GREY_D: str = "#444444"
GRAY_E: str = "#222222"
GREY_E: str = "#222222"
BLACK: str = "#000000"
LIGHTER_GRAY: str = "#DDDDDD"
LIGHTER_GREY: str = "#DDDDDD"
LIGHT_GRAY: str = "#BBBBBB"
LIGHT_GREY: str = "#BBBBBB"
GRAY: str = "#888888"
GREY: str = "#888888"
DARK_GRAY: str = "#444444"
DARK_GREY: str = "#444444"
DARKER_GRAY: str = "#222222"
DARKER_GREY: str = "#222222"
BLUE_A: str = "#C7E9F1"
BLUE_B: str = "#9CDCEB"
BLUE_C: str = "#58C4DD"
BLUE_D: str = "#29ABCA"
BLUE_E: str = "#236B8E"
PURE_BLUE: str = "#0000FF"
BLUE: str = "#58C4DD"
DARK_BLUE: str = "#236B8E"
TEAL_A: str = "#ACEAD7"
TEAL_B: str = "#76DDC0"
TEAL_C: str = "#5CD0B3"
TEAL_D: str = "#55C1A7"
TEAL_E: str = "#49A88F"
TEAL: str = "#5CD0B3"
GREEN_A: str = "#C9E2AE"
GREEN_B: str = "#A6CF8C"
GREEN_C: str = "#83C167"
GREEN_D: str = "#77B05D"
GREEN_E: str = "#699C52"
PURE_GREEN: str = "#00FF00"
GREEN: str = "#83C167"
YELLOW_A: str = "#FFF1B6"
YELLOW_B: str = "#FFEA94"
YELLOW_C: str = "#FFFF00"
YELLOW_D: str = "#F4D345"
YELLOW_E: str = "#E8C11C"
YELLOW: str = "#FFFF00"
GOLD_A: str = "#F7C797"
GOLD_B: str = "#F9B775"
GOLD_C: str = "#F0AC5F"
GOLD_D: str = "#E1A158"
GOLD_E: str = "#C78D46"
GOLD: str = "#F0AC5F"
RED_A: str = "#F7A1A3"
RED_B: str = "#FF8080"
RED_C: str = "#FC6255"
RED_D: str = "#E65A4C"
RED_E: str = "#CF5044"
PURE_RED: str = "#FF0000"
RED: str = "#FC6255"
MAROON_A: str = "#ECABC1"
MAROON_B: str = "#EC92AB"
MAROON_C: str = "#C55F73"
MAROON_D: str = "#A24D61"
MAROON_E: str = "#94424F"
MAROON: str = "#C55F73"
PURPLE_A: str = "#CAA3E8"
PURPLE_B: str = "#B189C6"
PURPLE_C: str = "#9A72AC"
PURPLE_D: str = "#715582"
PURPLE_E: str = "#644172"
PURPLE: str = "#9A72AC"
PINK: str = "#D147BD"
LIGHT_PINK: str = "#DC75CD"
ORANGE: str = "#FF862F"
LIGHT_BROWN: str = "#CD853F"
DARK_BROWN: str = "#8B4513"
GRAY_BROWN: str = "#736357"
GREY_BROWN: str = "#736357"

__all__ += [
    "WHITE",
    "GRAY_A",
    "GREY_A",
    "GRAY_B",
    "GREY_B",
    "GRAY_C",
    "GREY_C",
    "GRAY_D",
    "GREY_D",
    "GRAY_E",
    "GREY_E",
    "BLACK",
    "LIGHTER_GRAY",
    "LIGHTER_GREY",
    "LIGHT_GRAY",
    "LIGHT_GREY",
    "GRAY",
    "GREY",
    "DARK_GRAY",
    "DARK_GREY",
    "DARKER_GRAY",
    "DARKER_GREY",
    "BLUE_A",
    "BLUE_B",
    "BLUE_C",
    "BLUE_D",
    "BLUE_E",
    "PURE_BLUE",
    "BLUE",
    "DARK_BLUE",
    "TEAL_A",
    "TEAL_B",
    "TEAL_C",
    "TEAL_D",
    "TEAL_E",
    "TEAL",
    "GREEN_A",
    "GREEN_B",
    "GREEN_C",
    "GREEN_D",
    "GREEN_E",
    "PURE_GREEN",
    "GREEN",
    "YELLOW_A",
    "YELLOW_B",
    "YELLOW_C",
    "YELLOW_D",
    "YELLOW_E",
    "YELLOW",
    "GOLD_A",
    "GOLD_B",
    "GOLD_C",
    "GOLD_D",
    "GOLD_E",
    "GOLD",
    "RED_A",
    "RED_B",
    "RED_C",
    "RED_D",
    "RED_E",
    "PURE_RED",
    "RED",
    "MAROON_A",
    "MAROON_B",
    "MAROON_C",
    "MAROON_D",
    "MAROON_E",
    "MAROON",
    "PURPLE_A",
    "PURPLE_B",
    "PURPLE_C",
    "PURPLE_D",
    "PURPLE_E",
    "PURPLE",
    "PINK",
    "LIGHT_PINK",
    "ORANGE",
    "LIGHT_BROWN",
    "DARK_BROWN",
    "GRAY_BROWN",
    "GREY_BROWN",
]


def color_to_rgb(color: Color | str) -> np.ndarray:
    if isinstance(color, str):
        return hex_to_rgb(color)
    elif isinstance(color, Color):
        return np.array(color.get_rgb())
    else:
        raise ValueError("Invalid color type: " + str(color))


def color_to_rgba(color: Color | str, alpha: float = 1) -> np.ndarray:
    return np.array([*color_to_rgb(color), alpha])


def rgb_to_color(rgb: Iterable[float]) -> Color:
    return Color(rgb=rgb)


def rgba_to_color(rgba: Iterable[float]) -> Color:
    return rgb_to_color(rgba[:3])


def rgb_to_hex(rgb: Iterable[float]) -> str:
    return "#" + "".join("%02x" % round(255 * x) for x in rgb)


def hex_to_rgb(hex_code: str) -> np.ndarray:
    hex_part = hex_code[1:]
    if len(hex_part) == 3:
        hex_part = "".join([2 * c for c in hex_part])
    return np.array([int(hex_part[i : i + 2], 16) / 255 for i in range(0, 6, 2)])


def invert_color(color: Color) -> Color:
    return rgb_to_color(1.0 - color_to_rgb(color))


def color_to_int_rgb(color: Color) -> np.ndarray:
    return (255 * color_to_rgb(color)).astype("uint8")


def color_to_int_rgba(color: Color, opacity: float = 1.0) -> np.ndarray:
    alpha_multiplier = np.vectorize(lambda x: int(x * opacity))

    return alpha_multiplier(np.append(color_to_int_rgb(color), 255))


def color_gradient(
    reference_colors: Iterable[Color],
    length_of_output: int,
) -> list[Color]:
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


def interpolate_color(color1: Color, color2: Color, alpha: float) -> Color:
    rgb = interpolate(color_to_rgb(color1), color_to_rgb(color2), alpha)
    return rgb_to_color(rgb)


def average_color(*colors: Color) -> Color:
    rgbs = np.array(list(map(color_to_rgb, colors)))
    mean_rgb = np.apply_along_axis(np.mean, 0, rgbs)
    return rgb_to_color(mean_rgb)


def random_bright_color() -> Color:
    color = random_color()
    curr_rgb = color_to_rgb(color)
    new_rgb = interpolate(curr_rgb, np.ones(len(curr_rgb)), 0.5)
    return Color(rgb=new_rgb)


def random_color() -> Color:
    return random.choice([c.value for c in list(Colors)])


def get_shaded_rgb(
    rgb: np.ndarray,
    point: np.ndarray,
    unit_normal_vect: np.ndarray,
    light_source: np.ndarray,
) -> np.ndarray:
    to_sun = normalize(light_source - point)
    factor = 0.5 * np.dot(unit_normal_vect, to_sun) ** 3
    if factor < 0:
        factor *= 0.5
    result = rgb + factor
    return result
