"""Manim's (internal) color data structure and some utilities for
color conversion.

This module contains the implementation of :class:`.ManimColor`,
the data structure internally used to represent colors.

The preferred way of using these colors is by importing their constants from manim:

.. code-block:: pycon

    >>> from manim import RED, GREEN, BLUE
    >>> print(RED)
    #FC6255

Note this way uses the name of the colors in UPPERCASE.

.. note::

    The colors of type "C" have an alias equal to the colorname without a letter,
    e.g. GREEN = GREEN_C
"""

from __future__ import annotations

import colorsys

# logger = _config.logger
import random
import re
from typing import Any, Sequence, TypeVar, Union, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import Self, TypeAlias

from manim.typing import (
    HSV_Array_Float,
    HSV_Tuple_Float,
    ManimColorDType,
    ManimColorInternal,
    RGB_Array_Float,
    RGB_Array_Int,
    RGB_Tuple_Float,
    RGB_Tuple_Int,
    RGBA_Array_Float,
    RGBA_Array_Int,
    RGBA_Tuple_Float,
    RGBA_Tuple_Int,
)

from ...utils.space_ops import normalize

# import manim._config as _config

re_hex = re.compile("((?<=#)|(?<=0x))[A-F0-9]{6,8}", re.IGNORECASE)


class ManimColor:
    """Internal representation of a color.

    The ManimColor class is the main class for the representation of a color.
    It's internal representation is a 4 element array of floats corresponding
    to a [r,g,b,a] value where r,g,b,a can be between 0 to 1.

    This is done in order to reduce the amount of color inconsistencies by constantly
    casting between integers and floats which introduces errors.

    The class can accept any value of type :class:`ParsableManimColor` i.e.

    ManimColor, int, str, RGB_Tuple_Int, RGB_Tuple_Float, RGBA_Tuple_Int, RGBA_Tuple_Float, RGB_Array_Int,
    RGB_Array_Float, RGBA_Array_Int, RGBA_Array_Float

    ManimColor itself only accepts singular values and will directly interpret them into a single color if possible
    Be careful when passing strings to ManimColor it can create a big overhead for the color processing.

    If you want to parse a list of colors use the function :meth:`parse` in :class:`ManimColor` which assumes that
    you are going to pass a list of color so arrays will not be interpreted as a single color.

    .. warning::
        If you pass an array of numbers to :meth:`parse` it will interpret the r,g,b,a numbers in that array as colors
        so instead of the expect singular color you get and array with 4 colors.

    For conversion behaviors see the ``_internal`` functions for further documentation

    You can create a ``ManimColor`` instance via its classmethods. See the respective methods for more info.

    .. code-block:: python

        mycolor = ManimColor.from_rgb((0, 1, 0.4, 0.5))
        myothercolor = ManimColor.from_rgb((153, 255, 255))

    You can also convert between different color spaces:

    .. code-block:: python

        mycolor_hex = mycolor.to_hex()
        myoriginalcolor = ManimColor.from_hex(mycolor_hex).to_hsv()

    Parameters
    ----------
    value
        Some representation of a color (e.g., a string or
        a suitable tuple). The default ``None`` is ``BLACK``.
    alpha
        The opacity of the color. By default, colors are
        fully opaque (value 1.0).
    """

    def __init__(
        self,
        value: ParsableManimColor | None,
        alpha: float = 1.0,
    ) -> None:
        if value is None:
            self._internal_value = np.array((0, 0, 0, alpha), dtype=ManimColorDType)
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
                self._internal_value = ManimColor._internal_from_string(value)
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
        elif hasattr(value, "get_hex") and callable(value.get_hex):
            result = re_hex.search(value.get_hex())
            if result is None:
                raise ValueError(f"Failed to parse a color from {value}")

            self._internal_value = ManimColor._internal_from_hex_string(
                result.group(), alpha
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
        """Returns the internal value of the current Manim color [r,g,b,a] float array

        Returns
        -------
        ManimColorInternal
            internal color representation
        """
        return self.__value

    @_internal_value.setter
    def _internal_value(self, value: ManimColorInternal) -> None:
        """Overwrites the internal color value of the ManimColor object

        Parameters
        ----------
        value : ManimColorInternal
            The value which will overwrite the current color

        Raises
        ------
        TypeError
            Raises a TypeError if an invalid array is passed
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("value must be a numpy array")
        if value.shape[0] != 4:
            raise TypeError("Array must have 4 values exactly")
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

    # TODO: Maybe make 8 nibble hex also convertible ?
    @staticmethod
    def _internal_from_hex_string(hex: str, alpha: float) -> ManimColorInternal:
        """Internal function for converting a hex string into the internal representation of a ManimColor.

        .. warning::
            This does not accept any prefixes like # or similar in front of the hex string.
            This is just intended for the raw hex part

        *For internal use only*

        Parameters
        ----------
        hex : str
            hex string to be parsed
        alpha : float
            alpha value used for the color

        Returns
        -------
        ManimColorInternal
            Internal color representation
        """
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
        """Internal function for converting a rgb tuple of integers into the internal representation of a ManimColor.

        *For internal use only*

        Parameters
        ----------
        rgb : RGB_Tuple_Int
            integer rgb tuple to be parsed
        alpha : float, optional
            optional alpha value, by default 1.0

        Returns
        -------
        ManimColorInternal
            Internal color representation

        """
        value: np.ndarray = np.asarray(rgb, dtype=ManimColorDType).copy() / 255
        value.resize(4, refcheck=False)
        value[3] = alpha
        return value

    @staticmethod
    def _internal_from_rgb(
        rgb: RGB_Tuple_Float, alpha: float = 1.0
    ) -> ManimColorInternal:
        """Internal function for converting a rgb tuple of floats into the internal representation of a ManimColor.

        *For internal use only*

        Parameters
        ----------
        rgb : RGB_Tuple_Float
            float rgb tuple to be parsed

        alpha : float, optional
            optional alpha value, by default 1.0

        Returns
        -------
        ManimColorInternal
            Internal color representation
        """
        value: np.ndarray = np.asarray(rgb, dtype=ManimColorDType).copy()
        value.resize(4, refcheck=False)
        value[3] = alpha
        return value

    @staticmethod
    def _internal_from_int_rgba(rgba: RGBA_Tuple_Int) -> ManimColorInternal:
        """Internal function for converting a rgba tuple of integers into the internal representation of a ManimColor.

        *For internal use only*

        Parameters
        ----------
        rgba : RGBA_Tuple_Int
            int rgba tuple to be parsed

        Returns
        -------
        ManimColorInternal
            Internal color representation
        """
        return np.asarray(rgba, dtype=ManimColorDType) / 255

    @staticmethod
    def _internal_from_rgba(rgba: RGBA_Tuple_Float) -> ManimColorInternal:
        """Internal function for converting a rgba tuple of floats into the internal representation of a ManimColor.

        *For internal use only*

        Parameters
        ----------
        rgba : RGBA_Tuple_Float
            int rgba tuple to be parsed

        Returns
        -------
        ManimColorInternal
            Internal color representation
        """
        return np.asarray(rgba, dtype=ManimColorDType)

    @staticmethod
    def _internal_from_string(name: str) -> ManimColorInternal:
        """Internal function for converting a string into the internal representation of a ManimColor.
        This is not used for hex strings, please refer to :meth:`_internal_from_hex` for this functionality.

        *For internal use only*

        Parameters
        ----------
        name : str
            The color name to be parsed into a color. Refer to the different color Modules in the documentation Page to
            find the corresponding Color names.

        Returns
        -------
        ManimColorInternal
            Internal color representation

        Raises
        ------
        ValueError
            Raises a ValueError if the color name is not present with manim
        """
        from . import _all_color_dict

        upper_name = name.upper()

        if upper_name in _all_color_dict:
            return _all_color_dict[upper_name]._internal_value
        else:
            raise ValueError(f"Color {name} not found")

    def to_integer(self) -> int:
        """Converts the current ManimColor into an integer

        Returns
        -------
        int
            integer representation of the color

        .. warning::
            This will return only the rgb part of the color
        """
        return int.from_bytes(
            (self._internal_value[:3] * 255).astype(int).tobytes(), "big"
        )

    def to_rgb(self) -> RGB_Array_Float:
        """Converts the current ManimColor into a rgb array of floats

        Returns
        -------
        RGB_Array_Float
            rgb array with 3 elements of type float
        """
        return self._internal_value[:3]

    def to_int_rgb(self) -> RGB_Array_Int:
        """Converts the current ManimColor into a rgb array of int

        Returns
        -------
        RGB_Array_Int
            rgb array with 3 elements of type int
        """
        return (self._internal_value[:3] * 255).astype(int)

    def to_rgba(self) -> RGBA_Array_Float:
        """Converts the current ManimColor into a rgba array of floats

        Returns
        -------
        RGBA_Array_Float
            rgba array with 4 elements of type float
        """
        return self._internal_value

    def to_int_rgba(self) -> RGBA_Array_Int:
        """Converts the current ManimColor into a rgba array of int


        Returns
        -------
        RGBA_Array_Int
            rgba array with 4 elements of type int
        """
        return (self._internal_value * 255).astype(int)

    def to_rgba_with_alpha(self, alpha: float) -> RGBA_Array_Float:
        """Converts the current ManimColor into a rgba array of float as :meth:`to_rgba` but you can change the alpha
        value.

        Parameters
        ----------
        alpha : float
            alpha value to be used in the return value

        Returns
        -------
        RGBA_Array_Float
            rgba array with 4 elements of type float
        """
        return np.fromiter((*self._internal_value[:3], alpha), dtype=ManimColorDType)

    def to_int_rgba_with_alpha(self, alpha: float) -> RGBA_Array_Int:
        """Converts the current ManimColor into a rgba array of integers as :meth:`to_int_rgba` but you can change the alpha
        value.

        Parameters
        ----------
        alpha : float
            alpha value to be used for the return value. (Will automatically be scaled from 0-1 to 0-255 so just pass 0-1)

        Returns
        -------
        RGBA_Array_Int
            rgba array with 4 elements of type int
        """
        tmp = self._internal_value * 255
        tmp[3] = alpha * 255
        return tmp.astype(int)

    def to_hex(self, with_alpha: bool = False) -> str:
        """Converts the manim color to a hexadecimal representation of the color

        Parameters
        ----------
        with_alpha : bool, optional
            Changes the result from 6 to 8 values where the last 2 nibbles represent the alpha value of 0-255,
            by default False

        Returns
        -------
        str
            A hex string starting with a # with either 6 or 8 nibbles depending on your input, by default 6 i.e #XXXXXX
        """
        tmp = (
            f"#{int(self._internal_value[0] * 255):02X}"
            f"{int(self._internal_value[1] * 255):02X}"
            f"{int(self._internal_value[2] * 255):02X}"
        )
        if with_alpha:
            tmp += f"{int(self._internal_value[3] * 255):02X}"
        return tmp

    def to_hsv(self) -> HSV_Array_Float:
        """Converts the Manim Color to HSV array.

        .. note::
           Be careful this returns an array in the form `[h, s, v]` where the elements are floats.
           This might be confusing because rgb can also be an array of floats so you might want to annotate the usage
           of this function in your code by typing the variables with :class:`HSV_Array_Float` in order to differentiate
           between rgb arrays and hsv arrays

        Returns
        -------
        HSV_Array_Float
            A hsv array containing 3 elements of type float ranging from 0 to 1
        """
        return colorsys.rgb_to_hsv(*self.to_rgb())

    def invert(self, with_alpha=False) -> ManimColor:
        """Returns an linearly inverted version of the color (no inplace changes)

        Parameters
        ----------
        with_alpha : bool, optional
            if true the alpha value will be inverted too, by default False

            .. note::
                This can result in unintended behavior where objects are not displayed because their alpha
                value is suddenly 0 or very low. Please keep that in mind when setting this to true

        Returns
        -------
        ManimColor
            The linearly inverted ManimColor
        """
        return ManimColor(1.0 - self._internal_value, with_alpha)

    def interpolate(self, other: ManimColor, alpha: float) -> ManimColor:
        """Interpolates between the current and the given ManimColor an returns the interpolated color

        Parameters
        ----------
        other : ManimColor
            The other ManimColor to be used for interpolation
        alpha : float
            A point on the line in rgba colorspace connecting the two colors i.e. the interpolation point

            0 corresponds to the current ManimColor and 1 corresponds to the other ManimColor

        Returns
        -------
        ManimColor
            The interpolated ManimColor
        """
        return ManimColor(
            self._internal_value * (1 - alpha) + other._internal_value * alpha
        )

    @classmethod
    def from_rgb(
        cls,
        rgb: RGB_Array_Float | RGB_Tuple_Float | RGB_Array_Int | RGB_Tuple_Int,
        alpha: float = 1.0,
    ) -> Self:
        """Creates a ManimColor from an RGB Array. Automagically decides which type it is int/float

        .. warning::
            Please make sure that your elements are not floats if you want integers. A 5.0 will result in the input
            being interpreted as if it was a float rgb array with the value 5.0 and not the integer 5


        Parameters
        ----------
        rgb : RGB_Array_Float | RGB_Tuple_Float | RGB_Array_Int | RGB_Tuple_Int
            Any 3 Element Iterable
        alpha : float, optional
            alpha value to be used in the color, by default 1.0

        Returns
        -------
        ManimColor
            Returns the ManimColor object
        """
        return cls(rgb, alpha)

    @classmethod
    def from_rgba(
        cls, rgba: RGBA_Array_Float | RGBA_Tuple_Float | RGBA_Array_Int | RGBA_Tuple_Int
    ) -> Self:
        """Creates a ManimColor from an RGBA Array. Automagically decides which type it is int/float

        .. warning::
            Please make sure that your elements are not floats if you want integers. A 5.0 will result in the input
            being interpreted as if it was a float rgb array with the value 5.0 and not the integer 5

        Parameters
        ----------
        rgba : RGBA_Array_Float | RGBA_Tuple_Float | RGBA_Array_Int | RGBA_Tuple_Int
            Any 4 Element Iterable

        Returns
        -------
        ManimColor
            Returns the ManimColor object
        """
        return cls(rgba)

    @classmethod
    def from_hex(cls, hex: str, alpha: float = 1.0) -> Self:
        """Creates a Manim Color from a hex string, prefixes allowed # and 0x

        Parameters
        ----------
        hex : str
            The hex string to be converted (currently only supports 6 nibbles)
        alpha : float, optional
            alpha value to be used for the hex string, by default 1.0

        Returns
        -------
        ManimColor
            The ManimColor represented by the hex string
        """
        return cls(hex, alpha)

    @classmethod
    def from_hsv(
        cls, hsv: HSV_Array_Float | HSV_Tuple_Float, alpha: float = 1.0
    ) -> Self:
        """Creates a ManimColor from an HSV Array

        Parameters
        ----------
        hsv : HSV_Array_Float | HSV_Tuple_Float
            Any 3 Element Iterable containing floats from 0-1
        alpha : float, optional
            the alpha value to be used, by default 1.0

        Returns
        -------
        ManimColor
            The ManimColor with the corresponding RGB values to the HSV
        """
        rgb = colorsys.hsv_to_rgb(*hsv)
        return cls(rgb, alpha)

    @overload
    @classmethod
    def parse(
        cls,
        color: ParsableManimColor | None,
        alpha: float = ...,
    ) -> Self: ...

    @overload
    @classmethod
    def parse(
        cls,
        color: Sequence[ParsableManimColor],
        alpha: float = ...,
    ) -> list[Self]: ...

    @classmethod
    def parse(
        cls,
        color: ParsableManimColor | list[ParsableManimColor] | None,
        alpha: float = 1.0,
    ) -> Self | list[Self]:
        """
        Handles the parsing of a list of colors or a single color.

        Parameters
        ----------
        color
            The color or list of colors to parse. Note that this function can not accept rgba tuples. It will assume that you mean list[ManimColor] and will return a list of ManimColors.
        alpha
            The alpha value to use if a single color is passed. or if a list of colors is passed to set the value of all colors.

        Returns
        -------
        ManimColor
            Either a list of colors or a singular color depending on the input
        """
        if isinstance(color, (list, tuple)):
            return [cls(c, alpha) for c in color]  # type: ignore
        return cls(color, alpha)  # type: ignore

    @staticmethod
    def gradient(colors: list[ManimColor], length: int):
        """This is not implemented by now refer to :func:`color_gradient` for a working implementation for now"""
        # TODO: implement proper gradient, research good implementation for this or look at 3b1b implementation
        raise NotImplementedError

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
"""`ParsableManimColor` represents all the types which can be parsed
to a color in Manim.
"""


ManimColorT = TypeVar("ManimColorT", bound=ManimColor)


def color_to_rgb(color: ParsableManimColor) -> RGB_Array_Float:
    """Helper function for use in functional style programming.
    Refer to :meth:`to_rgb` in :class:`ManimColor`.

    Parameters
    ----------
    color : ParsableManimColor
        A color

    Returns
    -------
    RGB_Array_Float
        the corresponding rgb array
    """
    return ManimColor(color).to_rgb()


def color_to_rgba(color: ParsableManimColor, alpha: float = 1) -> RGBA_Array_Float:
    """Helper function for use in functional style programming refer to :meth:`to_rgba_with_alpha` in :class:`ManimColor`

    Parameters
    ----------
    color : ParsableManimColor
        A color
    alpha : float, optional
        alpha value to be used in the color, by default 1

    Returns
    -------
    RGBA_Array_Float
        the corresponding rgba array
    """
    return ManimColor(color).to_rgba_with_alpha(alpha)


def color_to_int_rgb(color: ParsableManimColor) -> RGB_Array_Int:
    """Helper function for use in functional style programming refer to :meth:`to_int_rgb` in :class:`ManimColor`

    Parameters
    ----------
    color : ParsableManimColor
        A color

    Returns
    -------
    RGB_Array_Int
        the corresponding int rgb array
    """
    return ManimColor(color).to_int_rgb()


def color_to_int_rgba(color: ParsableManimColor, alpha: float = 1.0) -> RGBA_Array_Int:
    """Helper function for use in functional style programming refer to :meth:`to_int_rgba_with_alpha` in :class:`ManimColor`

    Parameters
    ----------
    color : ParsableManimColor
        A color
    alpha : float, optional
        alpha value to be used in the color, by default 1.0

    Returns
    -------
    RGBA_Array_Int
        the corresponding int rgba array
    """
    return ManimColor(color).to_int_rgba_with_alpha(alpha)


def rgb_to_color(
    rgb: RGB_Array_Float | RGB_Tuple_Float | RGB_Array_Int | RGB_Tuple_Int,
) -> ManimColor:
    """Helper function for use in functional style programming refer to :meth:`from_rgb` in :class:`ManimColor`

    Parameters
    ----------
    rgb : RGB_Array_Float | RGB_Tuple_Float
        A 3 element iterable

    Returns
    -------
    ManimColor
        A ManimColor with the corresponding value
    """
    return ManimColor.from_rgb(rgb)


def rgba_to_color(
    rgba: RGBA_Array_Float | RGBA_Tuple_Float | RGBA_Array_Int | RGBA_Tuple_Int,
) -> ManimColor:
    """Helper function for use in functional style programming refer to :meth:`from_rgba` in :class:`ManimColor`

    Parameters
    ----------
    rgba : RGBA_Array_Float | RGBA_Tuple_Float
        A 4 element iterable

    Returns
    -------
    ManimColor
        A ManimColor with the corresponding value
    """
    return ManimColor.from_rgba(rgba)


def rgb_to_hex(
    rgb: RGB_Array_Float | RGB_Tuple_Float | RGB_Array_Int | RGB_Tuple_Int,
) -> str:
    """Helper function for use in functional style programming refer to :meth:`from_rgb` in :class:`ManimColor`

    Parameters
    ----------
    rgb : RGB_Array_Float | RGB_Tuple_Float
        A 3 element iterable

    Returns
    -------
    str
        A hex representation of the color, refer to :meth:`to_hex` in :class:`ManimColor`
    """
    return ManimColor.from_rgb(rgb).to_hex()


def hex_to_rgb(hex_code: str) -> RGB_Array_Float:
    """Helper function for use in functional style programming refer to :meth:`to_hex` in :class:`ManimColor`

    Parameters
    ----------
    hex_code : str
        A hex string representing a color

    Returns
    -------
    RGB_Array_Float
        RGB array representing the color
    """
    return ManimColor(hex_code).to_rgb()


def invert_color(color: ManimColorT) -> ManimColorT:
    """Helper function for use in functional style programming refer to :meth:`invert` in :class:`ManimColor`

    Parameters
    ----------
    color : ManimColor
        A ManimColor

    Returns
    -------
    ManimColor
        The linearly inverted ManimColor
    """
    return color.invert()


def interpolate_arrays(
    arr1: npt.NDArray[Any], arr2: npt.NDArray[Any], alpha: float
) -> np.ndarray:
    """Helper function used in Manim to fade between two objects smoothly

    Parameters
    ----------
    arr1 : npt.NDArray[Any]
        The first array of colors
    arr2 : npt.NDArray[Any]
        The second array of colors
    alpha : float
        The alpha value corresponding to the interpolation point between the two inputs

    Returns
    -------
    np.ndarray
        The interpolated value of the to arrays
    """
    return (1 - alpha) * arr1 + alpha * arr2


def color_gradient(
    reference_colors: Sequence[ParsableManimColor],
    length_of_output: int,
) -> list[ManimColor] | ManimColor:
    """Creates a list of colors interpolated between the input array of colors with a specific number of colors

    Parameters
    ----------
    reference_colors : Sequence[ParsableManimColor]
        The colors to be interpolated between or spread apart
    length_of_output : int
        The number of colors that the output should have, ideally more than the input

    Returns
    -------
    list[ManimColor] | ManimColor
        A list of ManimColor's which has the interpolated colors
    """
    if length_of_output == 0:
        return ManimColor(reference_colors[0])
    if len(reference_colors) == 1:
        return [ManimColor(reference_colors[0])] * length_of_output
    rgbs = [color_to_rgb(color) for color in reference_colors]
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
    color1: ManimColorT, color2: ManimColor, alpha: float
) -> ManimColorT:
    """Standalone function to interpolate two ManimColors and get the result refer to :meth:`interpolate` in :class:`ManimColor`

    Parameters
    ----------
    color1 : ManimColor
        First ManimColor
    color2 : ManimColor
        Second ManimColor
    alpha : float
        The alpha value determining the point of interpolation between the colors

    Returns
    -------
    ManimColor
        The interpolated ManimColor
    """
    return color1.interpolate(color2, alpha)


def average_color(*colors: ParsableManimColor) -> ManimColor:
    """Determines the Average color of the given parameters

    Returns
    -------
    ManimColor
        The average color of the input
    """
    rgbs = np.array([color_to_rgb(color) for color in colors])
    mean_rgb = np.apply_along_axis(np.mean, 0, rgbs)
    return rgb_to_color(mean_rgb)


def random_bright_color() -> ManimColor:
    """Returns you a random bright color

    .. warning::
        This operation is very expensive please keep in mind the performance loss.

    Returns
    -------
    ManimColor
        A bright ManimColor
    """
    curr_rgb = color_to_rgb(random_color())
    new_rgb = interpolate_arrays(curr_rgb, np.ones(len(curr_rgb)), 0.5)
    return ManimColor(new_rgb)


def random_color() -> ManimColor:
    """Return you a random ManimColor

    .. warning::
        This operation is very expensive please keep in mind the performance loss.

    Returns
    -------
    ManimColor
        _description_
    """
    import manim.utils.color.manim_colors as manim_colors

    return random.choice(manim_colors._all_manim_colors)


def get_shaded_rgb(
    rgb: npt.NDArray[Any],
    point: npt.NDArray[Any],
    unit_normal_vect: npt.NDArray[Any],
    light_source: npt.NDArray[Any],
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
