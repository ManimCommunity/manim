"""Colors and utility functions for conversion between different color models."""

from __future__ import annotations

# logger = _config.logger
import random
from typing import Any, Sequence, Union

import numpy as np
from typing_extensions import Literal, TypeAlias

from ...utils.space_ops import normalize

# import manim._config as _config


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

RGB_Array_Float: TypeAlias = "np.ndarray[Literal[3], np.dtype[ManimFloat]]"
RGB_Tuple_Float: TypeAlias = "tuple[float, float, float]"

RGB_Array_Int: TypeAlias = "np.ndarray[Literal[3], np.dtype[ManimInt]]"
RGB_Tuple_Int: TypeAlias = "tuple[int, int, int]"

RGBA_Array_Float: TypeAlias = "np.ndarray[Literal[4], np.dtype[ManimFloat]]"
RGBA_Tuple_Float: TypeAlias = "tuple[float, float, float, float]"

RGBA_Array_Int: TypeAlias = "np.ndarray[Literal[4], np.dtype[ManimInt]]"
RGBA_Tuple_Int: TypeAlias = "tuple[int, int, int, int]"

ManimColorInternal: TypeAlias = "np.ndarray[Literal[4], np.dtype[ManimColorDType]]"

import re

re_hex = re.compile("((?<=#)|(?<=0x))[A-F0-9]{6,8}", re.IGNORECASE)


class ManimColor:
    def __init__(
        self,
        value: ParsableManimColor,
        alpha: float = 1.0,
    ) -> None:
        if value is None:
            self._internal_value = np.array([0, 0, 0, alpha], dtype=ManimColorDType)
        elif isinstance(value, ManimColor):
            # logger.info(
            #     "ManimColor was passed another ManimColor. This is probably not what "
            #     "you want. Created a copy of the passed ManimColor instead."
            # )
            self._internal_value = value._internal_value
        elif isinstance(value, int):
            self._internal_value = ManimColor._internal_from_integer(value, alpha)
        elif isinstance(value, str):
            result = re_hex.search(value)
            if result is not None:
                self._internal_value = ManimColor._internal_from_hex_string(
                    result.group(), alpha
                )
            else:
                # This is not expected to be called on module initialization time
                # It can be horribly slow to convert a string to a color because
                # it has to access the dictionary of colors and find the right color
        elif isinstance(value, (list, tuple, np.ndarray)):
            length = len(value)
            if all(isinstance(x, float) for x in value):
                if length == 3:
                    self._internal_value = ManimColor._internal_from_rgb(value, alpha)  # type: ignore
                elif length == 4:
                    self._internal_value = ManimColor._internal_from_rgba(value)  # type: ignore
                else:
                    raise ValueError(
                        f"ManimColor only accepts lists/tuples/arrays of length 3 or 4, not {length}"
                    )
            else:
                if length == 3:
                    self._internal_value = ManimColor._internal_from_int_rgb(
                        value, alpha  # type: ignore
                    )
                elif length == 4:
                    self._internal_value = ManimColor._internal_from_int_rgba(value)  # type: ignore
                else:
                    raise ValueError(
                        f"ManimColor only accepts lists/tuples/arrays of length 3 or 4, not {length}"
                    )
        else:
            # logger.error(f"Invalid color value: {value}")
            raise TypeError(
                "ManimColor only accepts int, str, list[int, int, int], "
                "list[int, int, int, int], list[float, float, float], "
                f"list[float, float, float, float], not {type(value)}"
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
    def _internal_from_integer(value: int, alpha: float) -> ManimColorInternal:
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
    def _internal_from_hex_string(hex: str, alpha: float) -> ManimColorInternal:
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
    def _internal_from_int_rgb(
        rgb: RGB_Tuple_Int, alpha: float = 1.0
    ) -> ManimColorInternal:
        value: np.ndarray = np.asarray(rgb, dtype=ManimColorDType).copy() / 255
        value.resize(4, refcheck=False)
        value[3] = alpha
        return value

    @staticmethod
    def _internal_from_rgb(
        rgb: RGB_Tuple_Float, alpha: float = 1.0
    ) -> ManimColorInternal:
        value: np.ndarray = np.asarray(rgb, dtype=ManimColorDType).copy()
        value.resize(4, refcheck=False)
        value[3] = alpha
        return value

    @staticmethod
    def _internal_from_int_rgba(rgba: RGBA_Tuple_Int) -> ManimColorInternal:
        return np.asarray(rgba, dtype=ManimColorDType) / 255

    @staticmethod
    def _internal_from_rgba(rgba: RGBA_Tuple_Float) -> ManimColorInternal:
        return np.asarray(rgba, dtype=ManimColorDType)

    # TODO: This may be a bad idea but i don't know what else will be better without writing an endless list of colors
    @staticmethod
    def _internal_from_string(name: str) -> ManimColorInternal:
        from . import _all_color_dict

        if name.upper() in _all_color_dict:
            return _all_color_dict[name]._internal_value
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
        return ManimColor(1.0 - self._internal_value, with_alpha)

    def interpolate(self, other: ManimColor, alpha: float) -> ManimColor:
        return ManimColor(
            self._internal_value * (1 - alpha) + other._internal_value * alpha
        )

    @classmethod
    def from_rgb(
        cls,
        rgb: RGB_Array_Float | RGB_Tuple_Float | RGB_Array_Int | RGB_Tuple_Int,
        alpha: float = 1.0,
    ) -> ManimColor:
        return cls(rgb, alpha)

    @classmethod
    def from_rgba(
        cls, rgba: RGBA_Array_Float | RGBA_Tuple_Float | RGBA_Array_Int | RGBA_Tuple_Int
    ) -> ManimColor:
        return cls(rgba)

    @classmethod
    def from_hex(cls, hex: str, alpha: float = 1.0) -> ManimColor:
        return cls(hex, alpha)

    @classmethod
    def parse(
        cls,
        color: ParsableManimColor | list[ParsableManimColor] | None,
        alpha: float = 1.0,
    ) -> ManimColor | list[ManimColor]:
        """
        Handles the parsing of a list of colors or a single color.

        Parameters
        ----------
        color
            The color or list of colors to parse. Note that this function can not accept rgba tuples. It will assume that you mean list[ManimColor] and will return a list of ManimColors.
        alpha
            The alpha value to use if a single color is passed. or if a list of colors is passed to set the value of all colors.
        """
        if isinstance(color, (list, tuple)):
            return [cls(c, alpha) for c in color]  # type: ignore
        return cls(color, alpha)  # type: ignore

    @staticmethod
    def gradient(colors: list[ManimColor], length: int):
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.to_hex()}')"

    def __str__(self) -> str:
        return f"{self.to_hex()}"

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


ParsableManimColor: TypeAlias = Union[
    ManimColor,
    int,
    str,
    RGB_Tuple_Int,
    RGB_Tuple_Float,
    RGBA_Tuple_Int,
    RGBA_Tuple_Float,
    RGB_Array_Int,
    RGB_Array_Float,
    RGBA_Array_Int,
    RGBA_Array_Float,
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
    if len(reference_colors) == 1:
        return [ManimColor(reference_colors[0])] * length_of_output
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


def random_color() -> ManimColor:
    from . import _colors

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


__all__ = [
    "ManimColor",
    "ManimColorDType",
    "ParsableManimColor",
    "color_to_rgb",
    "color_to_rgba",
    "color_to_int_rgb",
    "color_to_int_rgba",
    "rgb_to_color",
    "rgba_to_color",
    "rgb_to_hex",
    "hex_to_rgb",
    "invert_color",
    "interpolate_arrays",
    "color_gradient",
    "interpolate_color",
    "average_color",
    "random_bright_color",
    "random_color",
    "get_shaded_rgb",
]
