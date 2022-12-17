"""Colors and utility functions for conversion between different color models."""

from __future__ import annotations

from typing import Any, Sequence

from typing_extensions import Annotated, Literal, TypeAlias

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

ManimColorDType: TypeAlias = np.float64
ManimFloat: TypeAlias = np.float64
ManimInt: TypeAlias = np.int64

RGB_Array_Float: TypeAlias = np.ndarray[Literal[3], np.dtype[ManimFloat]]
RGB_Tuple_Float: TypeAlias = tuple[float, float, float]

RGB_Array_Int: TypeAlias = np.ndarray[Literal[3], np.dtype[ManimInt]]
RGB_Tuple_Int: TypeAlias = tuple[int, int, int]

RGBA_Array_Float: TypeAlias = np.ndarray[Literal[4], np.dtype[ManimFloat]]
RGBA_Tuple_Float: TypeAlias = tuple[float, float, float, float]

RGBA_Array_Int: TypeAlias = np.ndarray[Literal[4], np.dtype[ManimInt]]
RGBA_Tuple_Int: TypeAlias = tuple[int, int, int, int]

ManimColorInternal: TypeAlias = np.ndarray[Literal[4], np.dtype[ManimColorDType]]


class ManimColor:
    def __init__(
        self,
        value: ParsableManimColor,
        alpha: float = 1.0,
        use_floats: bool = True,
    ) -> None:
        if value is None:
            self._internal_value = np.array([0, 0, 0, alpha], dtype=ManimColorDType)
        elif isinstance(value, ManimColor):
            # logger.warning(
            #     "ManimColor was passed another ManimColor. This is probably not what you want. Created a copy of the passed ManimColor instead."
            # )
            self._internal_value = value._internal_value
        elif isinstance(value, int):
            self._internal_value = ManimColor.internal_from_integer(value, alpha)
        elif isinstance(value, str):
            try:
                self._internal_value = ManimColor.internal_from_hex_string(value, alpha)
            except ValueError:
                self._internal_value = ManimColor.internal_from_string(value)
        elif isinstance(value, (list, tuple, np.ndarray)):
            length = len(value)
            if use_floats:
                if length == 3:
                    self._internal_value = ManimColor.internal_from_rgb(value, alpha)  # type: ignore
                elif length == 4:
                    self._internal_value = ManimColor.internal_from_rgba(value)  # type: ignore
                else:
                    raise ValueError(
                        f"ManimColor only accepts lists/tuples/arrays of length 3 or 4, not {length}"
                    )
            else:
                if length == 3:
                    self._internal_value = ManimColor.internal_from_int_rgb(
                        value, alpha  # type: ignore
                    )
                elif length == 4:
                    self._internal_value = ManimColor.internal_from_int_rgba(value)  # type: ignore
                else:
                    raise ValueError(
                        f"ManimColor only accepts lists/tuples/arrays of length 3 or 4, not {length}"
                    )
        else:
            # logger.error(f"Invalid color value: {value}")
            raise TypeError(
                f"ManimColor only accepts int, str, list[int, int, int], list[int, int, int, int], list[float, float, float], list[float, float, float, float], not {type(value)}"
            )

    @property
    def _internal_value(self) -> ManimColorInternal:
        return self.__value

    @_internal_value.setter
    def _internal_value(self, value: ManimColorInternal) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("value must be a numpy array")
        self.__value: ManimColorInternal = value

    @staticmethod
    def internal_from_integer(value: int, alpha: float) -> ManimColorInternal:
        return np.asarray(
            (
                ((value >> 16) & 0xFF) / 255,
                ((value >> 8) & 0xFF) / 255,
                ((value >> 0) & 0xFF) / 255,
                alpha,
            ),
            dtype=ManimColorDType,
        )

    @staticmethod
    def internal_from_hex_string(hex: str, alpha: float) -> ManimColorInternal:
        if hex.startswith("#"):
            hex = hex[1:]
        elif hex.startswith("0x"):
            hex = hex[2:]
        else:
            raise ValueError(f"Invalid hex value: {hex}")
        if len(hex) == 6:
            hex += "00"
        tmp = int(hex, 16)
        return np.asarray(
            (
                ((tmp >> 24) & 0xFF) / 255,
                ((tmp >> 16) & 0xFF) / 255,
                ((tmp >> 8) & 0xFF) / 255,
                alpha,
            ),
            dtype=ManimColorDType,
        )

    @staticmethod
    def internal_from_int_rgb(
        rgb: RGB_Tuple_Int, alpha: float = 1.0
    ) -> ManimColorInternal:
        value: np.ndarray = np.asarray(rgb, dtype=ManimColorDType).copy() / 255
        value.resize(4, refcheck=False)
        value[3] = alpha
        return value

    @staticmethod
    def internal_from_rgb(
        rgb: RGB_Tuple_Float, alpha: float = 1.0
    ) -> ManimColorInternal:
        value: np.ndarray = np.asarray(rgb, dtype=ManimColorDType).copy()
        value.resize(4, refcheck=False)
        value[3] = alpha
        return value

    @staticmethod
    def internal_from_int_rgba(rgba: RGBA_Tuple_Int) -> ManimColorInternal:
        return np.asarray(rgba, dtype=ManimColorDType) / 255

    @staticmethod
    def internal_from_rgba(rgba: RGBA_Tuple_Float) -> ManimColorInternal:
        return np.asarray(rgba, dtype=ManimColorDType)

    # TODO: This may be a bad idea but i don't know what else will be better without writing an endless list of colors
    @staticmethod
    def internal_from_string(name: str) -> ManimColorInternal:
        if name.upper() in globals():
            return globals()[name]._internal_value
        else:
            raise ValueError(f"Color {name} not found")

    def to_integer(self) -> int:
        return int.from_bytes(
            (self._internal_value[:3] * 255).astype(int).tobytes(), "big"
        )

    def to_rgb(self) -> RGB_Array_Float:
        return self._internal_value[:3]

    def to_int_rgb(self) -> RGB_Array_Int:
        return (self._internal_value[:3] * 255).astype(int)

    def to_rgba(self) -> RGBA_Array_Float:
        return self._internal_value

    def to_int_rgba(self) -> RGBA_Array_Int:
        return (self._internal_value * 255).astype(int)

    def to_rgba_with_alpha(self, alpha: float) -> RGBA_Array_Float:
        return np.fromiter((*self._internal_value[:3], alpha), dtype=ManimColorDType)

    # @deprecated("Use to_rgb_with_alpha instead.")
    def to_int_rgba_with_alpha(self, alpha: float) -> RGBA_Array_Int:
        tmp = self._internal_value * 255
        tmp[3] = alpha * 255
        return tmp.astype(int)

    def to_hex(self, with_alpha: bool = False) -> str:
        tmp = f"#{int(self._internal_value[0]*255):02X}{int(self._internal_value[1]*255):02X}{int(self._internal_value[2]*255):02X}"
        if with_alpha:
            tmp += f"{int(self._internal_value[3]*255):02X}"
        return tmp

    def invert(self, with_alpha=False) -> ManimColor:
        return ManimColor(1.0 - self._internal_value, with_alpha, use_floats=True)

    def interpolate(self, other: ManimColor, alpha: float) -> ManimColor:
        return ManimColor(
            self._internal_value * (1 - alpha) + other._internal_value * alpha
        )

    @classmethod
    def from_rgb(
        cls, rgb: RGB_Array_Float | RGB_Tuple_Float, alpha: float = 1.0
    ) -> ManimColor:
        return cls(rgb, alpha, use_floats=True)

    @classmethod
    def from_int_rgb(
        cls, rgb: RGB_Array_Int | RGB_Tuple_Int, alpha: float = 1.0
    ) -> ManimColor:
        return cls(rgb, alpha)

    @classmethod
    def from_rgba(cls, rgba: RGBA_Array_Float | RGBA_Tuple_Float) -> ManimColor:
        return cls(rgba, use_floats=True)

    @classmethod
    def from_int_rgba(cls, rgba: RGBA_Array_Int | RGBA_Tuple_Int) -> ManimColor:
        return cls(rgba)

    @classmethod
    def from_hex(cls, hex: str, alpha: float = 1.0) -> ManimColor:
        return cls(hex, alpha)

    @staticmethod
    def gradient(colors: list[ManimColor], length: int):
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_hex()})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.to_hex()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ManimColor):
            raise TypeError(
                f"Cannot compare {self.__class__.__name__} with {other.__class__.__name__}"
            )
        return np.allclose(self._internal_value, other._internal_value)

    def __add__(self, other: ManimColor) -> ManimColor:
        return ManimColor(self._internal_value + other._internal_value)

    def __sub__(self, other: ManimColor) -> ManimColor:
        return ManimColor(self._internal_value - other._internal_value)

    def __mul__(self, other: ManimColor) -> ManimColor:
        return ManimColor(self._internal_value * other._internal_value)

    def __truediv__(self, other: ManimColor) -> ManimColor:
        return ManimColor(self._internal_value / other._internal_value)

    def __floordiv__(self, other: ManimColor) -> ManimColor:
        return ManimColor(self._internal_value // other._internal_value)

    def __mod__(self, other: ManimColor) -> ManimColor:
        return ManimColor(self._internal_value % other._internal_value)

    def __pow__(self, other: ManimColor) -> ManimColor:
        return ManimColor(self._internal_value**other._internal_value)

    def __and__(self, other: ManimColor) -> ManimColor:
        return ManimColor(self.to_integer() & other.to_integer())

    def __or__(self, other: ManimColor) -> ManimColor:
        return ManimColor(self.to_integer() | other.to_integer())

    def __xor__(self, other: ManimColor) -> ManimColor:
        return ManimColor(self.to_integer() ^ other.to_integer())


ParsableManimColor: TypeAlias = (
    ManimColor
    | int
    | str
    | RGB_Tuple_Int
    | RGB_Tuple_Float
    | RGBA_Tuple_Int
    | RGBA_Tuple_Float
    | RGB_Array_Int
    | RGB_Array_Float
    | RGBA_Array_Int
    | RGBA_Array_Float
)

__all__ += ["ManimColor", "ParsableManimColor", "ManimColorDType"]

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


def color_to_rgb(color: ParsableManimColor) -> RGB_Array_Float:
    return ManimColor(color).to_rgb()


def color_to_rgba(color: ParsableManimColor, alpha: float = 1) -> RGBA_Array_Float:
    return ManimColor(color).to_rgba_with_alpha(alpha)


def color_to_int_rgb(color: ManimColor) -> RGB_Array_Int:
    return ManimColor(color).to_int_rgb()


def color_to_int_rgba(color: ManimColor, alpha: float = 1.0) -> RGBA_Array_Int:
    return ManimColor(color).to_int_rgba_with_alpha(alpha)


def rgb_to_color(rgb: RGB_Array_Float | RGB_Tuple_Float) -> ManimColor:
    return ManimColor.from_rgb(rgb)


def rgba_to_color(rgba: RGBA_Array_Float | RGBA_Tuple_Float) -> ManimColor:
    return ManimColor.from_rgba(rgba)


def rgb_to_hex(rgb: RGB_Array_Float | RGB_Tuple_Float) -> str:
    return ManimColor.from_rgb(rgb).to_hex()


def hex_to_rgb(hex_code: str) -> RGB_Array_Float:
    return ManimColor(hex_code).to_rgb()


def invert_color(color: ManimColor) -> ManimColor:
    return color.invert()


def interpolate_arrays(
    arr1: np.ndarray[Any, Any], arr2: np.ndarray[Any, Any], alpha: float
) -> np.ndarray:
    return (1 - alpha) * arr1 + alpha * arr2


def color_gradient(
    reference_colors: Sequence[ParsableManimColor],
    length_of_output: int,
) -> list[ManimColor] | ManimColor:
    if length_of_output == 0:
        return ManimColor(reference_colors[0])
    rgbs = list(map(color_to_rgb, reference_colors))
    alphas = np.linspace(0, (len(rgbs) - 1), length_of_output)
    floors = alphas.astype("int")
    alphas_mod1 = alphas % 1
    # End edge case
    alphas_mod1[-1] = 1
    floors[-1] = len(rgbs) - 2
    return [
        rgb_to_color((rgbs[i] * (1 - alpha)) + (rgbs[i + 1] * alpha))
        for i, alpha in zip(floors, alphas_mod1)
    ]


def interpolate_color(
    color1: ManimColor, color2: ManimColor, alpha: float
) -> ManimColor:
    return color1.interpolate(color2, alpha)


def average_color(*colors: ManimColor) -> ManimColor:
    rgbs = np.array(list(map(color_to_rgb, colors)))
    mean_rgb = np.apply_along_axis(np.mean, 0, rgbs)
    return rgb_to_color(mean_rgb)


def random_bright_color() -> ManimColor:
    color = random_color()
    curr_rgb = color_to_rgb(color)
    new_rgb = interpolate_arrays(curr_rgb, np.ones(len(curr_rgb)), 0.5)
    return ManimColor(new_rgb)


_colors: list[ManimColor] = [x for x in globals().values() if isinstance(x, ManimColor)]


def random_color() -> ManimColor:
    return random.choice(_colors)


def get_shaded_rgb(
    rgb: np.ndarray,
    point: np.ndarray,
    unit_normal_vect: np.ndarray,
    light_source: np.ndarray,
) -> RGBA_Array_Float:
    to_sun = normalize(light_source - point)
    factor = 0.5 * np.dot(unit_normal_vect, to_sun) ** 3
    if factor < 0:
        factor *= 0.5
    result = rgb + factor
    return result
