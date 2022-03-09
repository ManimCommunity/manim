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
import warnings
from enum import Enum, EnumMeta
from typing import Iterable

import numpy as np
from colour import Color, web2rgb

from ..utils.bezier import interpolate
from ..utils.space_ops import normalize


class ColorsMeta(EnumMeta):
    def __getattribute__(self, colorName):
        colors = [
            "white",
            "gray_a",
            "gray_b",
            "gray_c",
            "gray_d",
            "gray_e",
            "black",
            "lighter_gray",
            "gray",
            "darker_gray",
            "blue_a",
            "blue_b",
            "blue_c",
            "blue_d",
            "blue_e",
            "pure_blue",
            "blue",
            "dark_blue",
            "teal_a",
            "teal_b",
            "teal_c",
            "teal_d",
            "teal_e",
            "teal",
            "green_a",
            "green_b",
            "green_c",
            "green_d",
            "green_e",
            "pure_green",
            "green",
            "yellow_a",
            "yellow_b",
            "yellow_c",
            "yellow_d",
            "yellow_e",
            "yellow",
            "gold_a",
            "gold_b",
            "gold_c",
            "gold_d",
            "gold_e",
            "gold",
            "red_a",
            "red_b",
            "red_c",
            "red_d",
            "red_e",
            "pure_red",
            "red",
            "maroon_a",
            "maroon_b",
            "maroon_c",
            "maroon_d",
            "maroon_e",
            "maroon",
            "purple_a",
            "purple_b",
            "purple_c",
            "purple_d",
            "purple_e",
            "purple",
            "pink",
            "light_pink",
            "orange",
            "light_brown",
            "dark_brown",
            "gray_brown",
        ]
        if colorName in colors:
            warnings.warn(
                "Color enums is deprecated in favor of the constants. "
                "The constants can be accessed by importing manim.utils.color",
                DeprecationWarning,
                stacklevel=2,
            )
        return EnumMeta.__getattribute__(self, colorName)


class Colors(Enum, metaclass=ColorsMeta):
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


WHITE: Color = Color("#FFFFFF")
GRAY_A: Color = Color("#DDDDDD")
GREY_A: Color = Color("#DDDDDD")
GRAY_B: Color = Color("#BBBBBB")
GREY_B: Color = Color("#BBBBBB")
GRAY_C: Color = Color("#888888")
GREY_C: Color = Color("#888888")
GRAY_D: Color = Color("#444444")
GREY_D: Color = Color("#444444")
GRAY_E: Color = Color("#222222")
GREY_E: Color = Color("#222222")
BLACK: Color = Color("#000000")
LIGHTER_GRAY: Color = Color("#DDDDDD")
LIGHTER_GREY: Color = Color("#DDDDDD")
LIGHT_GRAY: Color = Color("#BBBBBB")
LIGHT_GREY: Color = Color("#BBBBBB")
GRAY: Color = Color("#888888")
GREY: Color = Color("#888888")
DARK_GRAY: Color = Color("#444444")
DARK_GREY: Color = Color("#444444")
DARKER_GRAY: Color = Color("#222222")
DARKER_GREY: Color = Color("#222222")
BLUE_A: Color = Color("#C7E9F1")
BLUE_B: Color = Color("#9CDCEB")
BLUE_C: Color = Color("#58C4DD")
BLUE_D: Color = Color("#29ABCA")
BLUE_E: Color = Color("#236B8E")
PURE_BLUE: Color = Color("#0000FF")
BLUE: Color = Color("#58C4DD")
DARK_BLUE: Color = Color("#236B8E")
TEAL_A: Color = Color("#ACEAD7")
TEAL_B: Color = Color("#76DDC0")
TEAL_C: Color = Color("#5CD0B3")
TEAL_D: Color = Color("#55C1A7")
TEAL_E: Color = Color("#49A88F")
TEAL: Color = Color("#5CD0B3")
GREEN_A: Color = Color("#C9E2AE")
GREEN_B: Color = Color("#A6CF8C")
GREEN_C: Color = Color("#83C167")
GREEN_D: Color = Color("#77B05D")
GREEN_E: Color = Color("#699C52")
PURE_GREEN: Color = Color("#00FF00")
GREEN: Color = Color("#83C167")
YELLOW_A: Color = Color("#FFF1B6")
YELLOW_B: Color = Color("#FFEA94")
YELLOW_C: Color = Color("#FFFF00")
YELLOW_D: Color = Color("#F4D345")
YELLOW_E: Color = Color("#E8C11C")
YELLOW: Color = Color("#FFFF00")
GOLD_A: Color = Color("#F7C797")
GOLD_B: Color = Color("#F9B775")
GOLD_C: Color = Color("#F0AC5F")
GOLD_D: Color = Color("#E1A158")
GOLD_E: Color = Color("#C78D46")
GOLD: Color = Color("#F0AC5F")
RED_A: Color = Color("#F7A1A3")
RED_B: Color = Color("#FF8080")
RED_C: Color = Color("#FC6255")
RED_D: Color = Color("#E65A4C")
RED_E: Color = Color("#CF5044")
PURE_RED: Color = Color("#FF0000")
RED: Color = Color("#FC6255")
MAROON_A: Color = Color("#ECABC1")
MAROON_B: Color = Color("#EC92AB")
MAROON_C: Color = Color("#C55F73")
MAROON_D: Color = Color("#A24D61")
MAROON_E: Color = Color("#94424F")
MAROON: Color = Color("#C55F73")
PURPLE_A: Color = Color("#CAA3E8")
PURPLE_B: Color = Color("#B189C6")
PURPLE_C: Color = Color("#9A72AC")
PURPLE_D: Color = Color("#715582")
PURPLE_E: Color = Color("#644172")
PURPLE: Color = Color("#9A72AC")
PINK: Color = Color("#D147BD")
LIGHT_PINK: Color = Color("#DC75CD")
ORANGE: Color = Color("#FF862F")
LIGHT_BROWN: Color = Color("#CD853F")
DARK_BROWN: Color = Color("#8B4513")
GRAY_BROWN: Color = Color("#736357")
GREY_BROWN: Color = Color("#736357")

ALL_COLORS = {
    "WHITE": WHITE,
    "GRAY_A": GRAY_A,
    "GREY_A": GREY_B,
    "GRAY_B": GRAY_B,
    "GREY_B": GREY_B,
    "GRAY_C": GRAY_C,
    "GREY_C": GREY_C,
    "GRAY_D": GRAY_D,
    "GREY_D": GREY_D,
    "GRAY_E": GRAY_E,
    "GREY_E": GREY_E,
    "BLACK": BLACK,
    "LIGHTER_GRAY": LIGHTER_GRAY,
    "LIGHTER_GREY": LIGHTER_GREY,
    "LIGHT_GRAY": LIGHT_GRAY,
    "LIGHT_GREY": LIGHT_GREY,
    "GRAY": GRAY,
    "GREY": GREY,
    "DARK_GRAY": DARK_GRAY,
    "DARK_GREY": DARK_GREY,
    "DARKER_GRAY": DARKER_GRAY,
    "DARKER_GREY": DARKER_GREY,
    "BLUE_A": BLUE_A,
    "BLUE_B": BLUE_B,
    "BLUE_C": BLUE_C,
    "BLUE_D": BLUE_D,
    "BLUE_E": BLUE_E,
    "PURE_BLUE": PURE_BLUE,
    "BLUE": BLUE,
    "DARK_BLUE": DARK_BLUE,
    "TEAL_A": TEAL_A,
    "TEAL_B": TEAL_B,
    "TEAL_C": TEAL_C,
    "TEAL_D": TEAL_D,
    "TEAL_E": TEAL_E,
    "TEAL": TEAL,
    "GREEN_A": GREEN_A,
    "GREEN_B": GREEN_B,
    "GREEN_C": GREEN_C,
    "GREEN_D": GREEN_D,
    "GREEN_E": GREEN_E,
    "PURE_GREEN": PURE_GREEN,
    "GREEN": GREEN,
    "YELLOW_A": YELLOW_A,
    "YELLOW_B": YELLOW_B,
    "YELLOW_C": YELLOW_C,
    "YELLOW_D": YELLOW_D,
    "YELLOW_E": YELLOW_E,
    "YELLOW": YELLOW,
    "GOLD_A": GOLD_A,
    "GOLD_B": GOLD_B,
    "GOLD_C": GOLD_C,
    "GOLD_D": GOLD_D,
    "GOLD_E": GOLD_E,
    "GOLD": GOLD,
    "RED_A": RED_A,
    "RED_B": RED_B,
    "RED_C": RED_C,
    "RED_D": RED_D,
    "RED_E": RED_E,
    "PURE_RED": PURE_RED,
    "RED": RED,
    "MAROON_A": MAROON_A,
    "MAROON_B": MAROON_B,
    "MAROON_C": MAROON_C,
    "MAROON_D": MAROON_D,
    "MAROON_E": MAROON_E,
    "MAROON": MAROON,
    "PURPLE_A": PURPLE_A,
    "PURPLE_B": PURPLE_B,
    "PURPLE_C": PURPLE_C,
    "PURPLE_D": PURPLE_D,
    "PURPLE_E": PURPLE_E,
    "PURPLE": PURPLE,
    "PINK": PINK,
    "LIGHT_PINK": LIGHT_PINK,
    "ORANGE": ORANGE,
    "LIGHT_BROWN": LIGHT_BROWN,
    "DARK_BROWN": DARK_BROWN,
    "GRAY_BROWN": GRAY_BROWN,
    "GREY_BROWN": GREY_BROWN,
}

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
    "ALL_COLORS",
]


def color_to_rgb(color: Color | str) -> np.ndarray:
    warnings.warn(
        "This method is not guaranteed to stay around. "
        "Please refer to colour module `Color.get_rgb` for Color to rgb conversion",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(color, str):
        return hex_to_rgb(color)
    elif isinstance(color, Color):
        return np.array(color.get_rgb())
    else:
        raise ValueError("Invalid color type: " + str(color))


def color_to_rgba(color: Color | str, alpha: float = 1) -> np.ndarray:
    if isinstance(color, str):
        return np.array([*web2rgb(color), alpha])
    elif isinstance(color, Color):
        return np.array([*color.get_rgb(), alpha])


def rgb_to_color(rgb: Iterable[float]) -> Color:
    return Color(rgb=rgb)


def rgba_to_color(rgba: Iterable[float]) -> Color:
    return rgb_to_color(rgba[:3])


def rgb_to_hex(rgb: Iterable[float]) -> str:
    warnings.warn(
        "This method is not guaranteed to stay around. "
        "Please refer to colour module `rgb2hex` for rgb to hex conversion",
        DeprecationWarning,
        stacklevel=2,
    )
    return "#" + "".join("%02x" % round(255 * x) for x in rgb)


def hex_to_rgb(hex_code: str) -> np.ndarray:
    warnings.warn(
        "This method is not guaranteed to stay around. "
        "Please refer to colour module `hex2rgb` for hex to rgb conversion",
        DeprecationWarning,
        stacklevel=2,
    )
    hex_part = hex_code[1:]
    if len(hex_part) == 3:
        hex_part = "".join([2 * c for c in hex_part])
    return np.array([int(hex_part[i : i + 2], 16) / 255 for i in range(0, 6, 2)])


def invert_color(color: Color) -> Color:
    return rgb_to_color(1.0 - np.array(color.get_rgb()))


def color_to_int_rgb(color: Color | str) -> np.ndarray:
    if isinstance(color, str):
        return [round(i) for i in (255 * np.array(Color(color).get_rgb()))]
    elif isinstance(color, Color):
        return [round(i) for i in (255 * np.array(color.get_rgb()))]


def color_to_int_rgba(color: Color, opacity: float = 1.0) -> np.ndarray:

    alpha_multiplier = np.vectorize(lambda x: int(x * opacity))

    return alpha_multiplier(np.append(color_to_int_rgb(color), 255))


def color_gradient(
    reference_colors: Color | Iterable[Color],
    length_of_output: int,
) -> list[Color]:
    if length_of_output == 0:
        return reference_colors[0]
    if isinstance(reference_colors, Color):
        return list(reference_colors.range_to(reference_colors, length_of_output))
    rgbs = [np.array(c.get_rgb()) for c in reference_colors]
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
    rgb = interpolate(np.array(color1.get_rgb()), np.array(color2.get_rgb()), alpha)
    return rgb_to_color(rgb)


def average_color(*colors: Color) -> Color:
    rgbs = np.array([c.get_rgb() for c in colors])
    mean_rgb = np.apply_along_axis(np.mean, 0, rgbs)
    return rgb_to_color(mean_rgb)


def random_bright_color() -> Color:
    color = random_color()
    curr_rgb = color.get_rgb()
    new_rgb = interpolate(curr_rgb, np.ones(len(curr_rgb)), 0.5)
    return Color(rgb=new_rgb)


def random_color() -> Color:
    return random.choice(list(ALL_COLORS.values()))


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
