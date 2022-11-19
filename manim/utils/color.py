"""Colors and utility functions for conversion between different color models."""

from __future__ import annotations

from enum import Enum
from typing import Iterable, TypeAlias, TypedDict

# from manim._config import logger

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

import numpy as np

from ..utils.bezier import interpolate
from ..utils.space_ops import normalize

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


class ManimColor:
    def __init__(
        self,
        value: str
        | int
        | list[int, int, int]
        | list[int, int, int, int]
        | list[float, float, float]
        | list[float, float, float, float],
        alpha: float = 1.0,
        use_floats: bool = True,
    ) -> None:
        if isinstance(value, ManimColor):
            # logger.warning(
            #     "ManimColor was passed another ManimColor. This is probably not what you want. Created a copy of the passed ManimColor instead."
            # )
            self.value = value.value
        elif isinstance(value, int):
            self.value: int = value << 8 | int(alpha * 255)
        elif isinstance(value, str):
            self.value: int = ManimColor.int_from_hex(value, alpha)
        elif (
            isinstance(value, list)
            or isinstance(value, tuple)
            or isinstance(value, np.ndarray)
        ):
            length = len(value)

            if use_floats:
                if length == 3:
                    self.value: int = ManimColor.int_from_rgb(value, alpha)
                elif length == 4:
                    self.value: int = ManimColor.int_from_rgba(value)
            else:
                if length == 3:
                    self.value: int = ManimColor.int_from_int_rgb(value, alpha)
                elif length == 4:
                    self.value: int = ManimColor.int_from_int_rgba(value)
        else:
            # logger.error(f"Invalid color value: {value}")
            raise TypeError(
                f"ManimColor only accepts int, str, list[int, int, int], list[int, int, int, int], list[float, float, float], list[float, float, float, float], not {type(value)}"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_hex()})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.to_hex()})"

    def __eq__(self, other: ManimColor) -> bool:
        return self.value == other.value

    def __add__(self, other: ManimColor) -> ManimColor:
        return ManimColor(self.value + other.value)

    def __sub__(self, other: ManimColor) -> ManimColor:
        return ManimColor(self.value - other.value)

    def __mul__(self, other: ManimColor) -> ManimColor:
        return ManimColor(self.value * other.value)

    def __truediv__(self, other: ManimColor) -> ManimColor:
        return ManimColor(self.value / other.value)

    def __floordiv__(self, other: ManimColor) -> ManimColor:
        return ManimColor(self.value // other.value)

    def __mod__(self, other: ManimColor) -> ManimColor:
        return ManimColor(self.value % other.value)

    def __pow__(self, other: ManimColor) -> ManimColor:
        return ManimColor(self.value**other.value)

    def __and__(self, other: ManimColor) -> ManimColor:
        return ManimColor(self.value & other.value)

    def __or__(self, other: ManimColor) -> ManimColor:
        return ManimColor(self.value | other.value)

    def __xor__(self, other: ManimColor) -> ManimColor:
        return ManimColor(self.value ^ other.value)

    def to_rgb(self) -> np.ndarray:
        return self.to_int_rgb() / 255

    def to_int_rgb(self) -> list[int, int, int]:
        return np.array(
            [
                (self.value >> 24) & 0xFF,
                (self.value >> 16) & 0xFF,
                (self.value >> 8) & 0xFF,
            ]
        )

    def to_rgba(self) -> np.ndarray:
        return self.to_int_rgba() / 255

    def to_int_rgba(self) -> list[int, int, int, int]:
        return np.array(
            [
                (self.value >> 24) & 0xFF,
                (self.value >> 16) & 0xFF,
                (self.value >> 8) & 0xFF,
                self.value & 0xFF,
            ]
        )

    def to_rgb_with_alpha(self, alpha: float) -> list[float, float, float, float]:
        return self.to_int_rgb_with_alpha(alpha) / 255

    def to_int_rgb_with_alpha(self, alpha: float) -> list[int, int, int, int]:
        return np.array(
            [
                (self.value >> 24) & 0xFF,
                (self.value >> 16) & 0xFF,
                (self.value >> 8) & 0xFF,
                int(alpha * 255),
            ]
        )

    def to_hex(self) -> str:
        return f"#{self.value:08X}"

    @staticmethod
    def int_from_int_rgb(rgb: list[int, int, int], alpha: float = 1.0) -> ManimColor:
        return rgb[0] << 24 | rgb[1] << 16 | rgb[2] << 8 | int(alpha * 255)

    @staticmethod
    def int_from_rgb(rgb: list[float, float, float], alpha: float = 1.0) -> ManimColor:
        return ManimColor.int_from_int_rgb((np.asarray(rgb) * 255).astype(int), alpha)

    @staticmethod
    def int_from_int_rgba(rgba: list[int, int, int, int]) -> ManimColor:
        return rgba[0] << 24 | rgba[1] << 16 | rgba[2] << 8 | rgba[3]

    @staticmethod
    def int_from_rgba(rgba: list[float, float, float, float]) -> ManimColor:
        return ManimColor.from_int_rgba((np.asarray(rgba) * 255).astype(int))

    @staticmethod
    def int_from_hex(hex: str, alpha: float) -> ManimColor:
        if hex.startswith("#"):
            hex = hex[1:]
        if hex.startswith("0x"):
            hex = hex[2:]
        if len(hex) == 6:
            hex += "00"
        return int(hex, 16) | int(alpha * 255)

    def invert(self, with_alpha=False) -> ManimColor:
        return ManimColor(0xFFFFFFFF - (self.value & 0xFFFFFFFF))

    @classmethod
    def from_rgb(cls, rgb: list[float, float, float], alpha: float = 1.0) -> ManimColor:
        return cls(rgb, alpha, use_floats=True)

    @classmethod
    def from_int_rgb(cls, rgb: list[int, int, int], alpha: float = 1.0) -> ManimColor:
        return cls(rgb, alpha)

    @classmethod
    def from_rgba(cls, rgba: list[float, float, float, float]) -> ManimColor:
        return cls(rgba, use_floats=True)

    @classmethod
    def from_int_rgba(cls, rgba: list[int, int, int, int]) -> ManimColor:
        return cls(rgba)

    @classmethod
    def from_hex(cls, hex: str, alpha: float = 1.0) -> ManimColor:
        return cls(hex, alpha)

    @staticmethod
    def gradient(colors: list[ManimColor], length: int):
        ...

    @classmethod
    def parse(cls, value, alpha=1.0, use_floats=True) -> ManimColor:
        if isinstance(value, ManimColor):
            return value
        return cls(value, alpha, use_floats)


ParsableManimColor: TypeAlias = (
    ManimColor
    | int
    | str
    | list[float, float, float]
    | list[float, float, float, float]
    | list[int, int, int]
    | list[int, int, int, int]
)
__all__ += ["ManimColor", "ParsableManimColor"]

WHITE: ManimColor = ManimColor("#FFFFFF")
GRAY_A: ManimColor = ManimColor("#DDDDDD")
GREY_A: ManimColor = ManimColor("#DDDDDD")
GRAY_B: ManimColor = ManimColor("#BBBBBB")
GREY_B: ManimColor = ManimColor("#BBBBBB")
GRAY_C: ManimColor = ManimColor("#888888")
GREY_C: ManimColor = ManimColor("#888888")
GRAY_D: ManimColor = ManimColor("#444444")
GREY_D: ManimColor = ManimColor("#444444")
GRAY_E: ManimColor = ManimColor("#222222")
GREY_E: ManimColor = ManimColor("#222222")
BLACK: ManimColor = ManimColor("#000000")
LIGHTER_GRAY: ManimColor = ManimColor("#DDDDDD")
LIGHTER_GREY: ManimColor = ManimColor("#DDDDDD")
LIGHT_GRAY: ManimColor = ManimColor("#BBBBBB")
LIGHT_GREY: ManimColor = ManimColor("#BBBBBB")
GRAY: ManimColor = ManimColor("#888888")
GREY: ManimColor = ManimColor("#888888")
DARK_GRAY: ManimColor = ManimColor("#444444")
DARK_GREY: ManimColor = ManimColor("#444444")
DARKER_GRAY: ManimColor = ManimColor("#222222")
DARKER_GREY: ManimColor = ManimColor("#222222")
BLUE_A: ManimColor = ManimColor("#C7E9F1")
BLUE_B: ManimColor = ManimColor("#9CDCEB")
BLUE_C: ManimColor = ManimColor("#58C4DD")
BLUE_D: ManimColor = ManimColor("#29ABCA")
BLUE_E: ManimColor = ManimColor("#236B8E")
PURE_BLUE: ManimColor = ManimColor("#0000FF")
BLUE: ManimColor = ManimColor("#58C4DD")
DARK_BLUE: ManimColor = ManimColor("#236B8E")
TEAL_A: ManimColor = ManimColor("#ACEAD7")
TEAL_B: ManimColor = ManimColor("#76DDC0")
TEAL_C: ManimColor = ManimColor("#5CD0B3")
TEAL_D: ManimColor = ManimColor("#55C1A7")
TEAL_E: ManimColor = ManimColor("#49A88F")
TEAL: ManimColor = ManimColor("#5CD0B3")
GREEN_A: ManimColor = ManimColor("#C9E2AE")
GREEN_B: ManimColor = ManimColor("#A6CF8C")
GREEN_C: ManimColor = ManimColor("#83C167")
GREEN_D: ManimColor = ManimColor("#77B05D")
GREEN_E: ManimColor = ManimColor("#699C52")
PURE_GREEN: ManimColor = ManimColor("#00FF00")
GREEN: ManimColor = ManimColor("#83C167")
YELLOW_A: ManimColor = ManimColor("#FFF1B6")
YELLOW_B: ManimColor = ManimColor("#FFEA94")
YELLOW_C: ManimColor = ManimColor("#FFFF00")
YELLOW_D: ManimColor = ManimColor("#F4D345")
YELLOW_E: ManimColor = ManimColor("#E8C11C")
YELLOW: ManimColor = ManimColor("#FFFF00")
GOLD_A: ManimColor = ManimColor("#F7C797")
GOLD_B: ManimColor = ManimColor("#F9B775")
GOLD_C: ManimColor = ManimColor("#F0AC5F")
GOLD_D: ManimColor = ManimColor("#E1A158")
GOLD_E: ManimColor = ManimColor("#C78D46")
GOLD: ManimColor = ManimColor("#F0AC5F")
RED_A: ManimColor = ManimColor("#F7A1A3")
RED_B: ManimColor = ManimColor("#FF8080")
RED_C: ManimColor = ManimColor("#FC6255")
RED_D: ManimColor = ManimColor("#E65A4C")
RED_E: ManimColor = ManimColor("#CF5044")
PURE_RED: ManimColor = ManimColor("#FF0000")
RED: ManimColor = ManimColor("#FC6255")
MAROON_A: ManimColor = ManimColor("#ECABC1")
MAROON_B: ManimColor = ManimColor("#EC92AB")
MAROON_C: ManimColor = ManimColor("#C55F73")
MAROON_D: ManimColor = ManimColor("#A24D61")
MAROON_E: ManimColor = ManimColor("#94424F")
MAROON: ManimColor = ManimColor("#C55F73")
PURPLE_A: ManimColor = ManimColor("#CAA3E8")
PURPLE_B: ManimColor = ManimColor("#B189C6")
PURPLE_C: ManimColor = ManimColor("#9A72AC")
PURPLE_D: ManimColor = ManimColor("#715582")
PURPLE_E: ManimColor = ManimColor("#644172")
PURPLE: ManimColor = ManimColor("#9A72AC")
PINK: ManimColor = ManimColor("#D147BD")
LIGHT_PINK: ManimColor = ManimColor("#DC75CD")
ORANGE: ManimColor = ManimColor("#FF862F")
LIGHT_BROWN: ManimColor = ManimColor("#CD853F")
DARK_BROWN: ManimColor = ManimColor("#8B4513")
GRAY_BROWN: ManimColor = ManimColor("#736357")
GREY_BROWN: ManimColor = ManimColor("#736357")

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


def color_to_rgb(color: ParsableManimColor) -> np.ndarray:
    if isinstance(color, ManimColor):
        return color.to_rgb()
    else:
        return ManimColor(color).to_rgb()


def color_to_rgba(color: ParsableManimColor, alpha: float = 1) -> np.ndarray:
    return np.array([*color_to_rgb(color), alpha])


def rgb_to_color(rgb: Iterable[float]) -> ManimColor:
    return ManimColor(rgb=rgb)


def rgba_to_color(rgba: Iterable[float]) -> ManimColor:
    return rgb_to_color(rgba[:3], alpha=rgba[3])


def rgb_to_hex(rgb: Iterable[float]) -> str:
    return "#" + "".join("%02x" % round(255 * x) for x in rgb)


def hex_to_rgb(hex_code: str) -> np.ndarray:
    hex_part = hex_code[1:]
    if len(hex_part) == 3:
        hex_part = "".join([2 * c for c in hex_part])
    return np.array([int(hex_part[i : i + 2], 16) / 255 for i in range(0, 6, 2)])


def invert_color(color: ManimColor) -> ManimColor:
    return rgb_to_color(1.0 - color_to_rgb(color))


def color_to_int_rgb(color: ManimColor) -> np.ndarray:
    return (255 * color_to_rgb(color)).astype("uint8")


def color_to_int_rgba(color: ManimColor, opacity: float = 1.0) -> np.ndarray:
    alpha_multiplier = np.vectorize(lambda x: int(x * opacity))

    return alpha_multiplier(np.append(color_to_int_rgb(color), 255))


def color_gradient(
    reference_colors: Iterable[ManimColor],
    length_of_output: int,
) -> list[ManimColor]:
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


def interpolate_color(
    color1: ManimColor, color2: ManimColor, alpha: float
) -> ManimColor:
    rgb = interpolate(color_to_rgb(color1), color_to_rgb(color2), alpha)
    return rgb_to_color(rgb)


def average_color(*colors: ManimColor) -> ManimColor:
    rgbs = np.array(list(map(color_to_rgb, colors)))
    mean_rgb = np.apply_along_axis(np.mean, 0, rgbs)
    return rgb_to_color(mean_rgb)


def random_bright_color() -> ManimColor:
    color = random_color()
    curr_rgb = color_to_rgb(color)
    new_rgb = interpolate(curr_rgb, np.ones(len(curr_rgb)), 0.5)
    return ManimColor(new_rgb)


_colors = [x for x in globals().values() if isinstance(x, ManimColor)]


def random_color() -> ManimColor:
    return random.choice([c.value for c in list(_colors)])


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
