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

===================
Custom Color Spaces
===================

Hello dear visitor, you seem to be interested in implementing a custom color class for a color space we don't currently support.

The current system is using a few indirections for ensuring a consistent behavior with all other color types in manim.

To implement a custom color space you must subclass :class:`ManimColor` and implement three important functions

:attr:`~.ManimColor._internal_value` is an ``@property`` implemented on :class:`ManimColor` with the goal of keeping a consistent internal representation that can be referenced by other functions in :class:`ManimColor`.
The getter should always return a value in the format of ``[r,g,b,a]`` as a numpy array which is in accordance with the type :class:`.ManimColorInternal`.
The setter should always accept a value in the format ``[r,g,b,a]`` which can be converted to whatever attributes you need.
This property acts as a proxy to whatever representation you need in your class.

:attr:`~ManimColor._internal_space` this is a readonly ``@property`` implemented on :class:`ManimColor` with the goal of a useful representation that can be used by operators and interpolation and color transform functions.
The only constraints on this value are that it needs to be a numpy array and the last value must be the opacity in a range ``0.0`` to ``1.0``.
Additionally your ``__init__`` must support this format as initialization value without additional parameters to ensure correct functionality of all other methods in :class:`ManimColor`.

:func:`~ManimColor._from_internal` is a ``@classmethod`` that converts an ``[r,g,b,a]`` value into suitable parameters for your ``__init__`` method and calls the cls parameter.
"""

from __future__ import annotations

import colorsys

# logger = _config.logger
import random
import re
from collections.abc import Sequence
from typing import Any, TypeVar, Union, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import Self, TypeAlias, TypeGuard, override

from manim.typing import (
    HSL_Array_Float,
    HSL_Tuple_Float,
    HSV_Array_Float,
    HSV_Tuple_Float,
    HSVA_Array_Float,
    HSVA_Tuple_Float,
    ManimColorDType,
    ManimColorInternal,
    ManimFloat,
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
                self._internal_value = ManimColor._internal_from_string(value, alpha)
        elif isinstance(value, (list, tuple, np.ndarray)):
            length = len(value)
            if all(isinstance(x, float) for x in value):
                if length == 3:
                    self._internal_value = ManimColor._internal_from_rgb(value, alpha)  # type: ignore[arg-type]
                elif length == 4:
                    self._internal_value = ManimColor._internal_from_rgba(value)  # type: ignore[arg-type]
                else:
                    raise ValueError(
                        f"ManimColor only accepts lists/tuples/arrays of length 3 or 4, not {length}"
                    )
            else:
                if length == 3:
                    self._internal_value = ManimColor._internal_from_int_rgb(
                        value,  # type: ignore[arg-type]
                        alpha,
                    )
                elif length == 4:
                    self._internal_value = ManimColor._internal_from_int_rgba(value)  # type: ignore[arg-type]
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
    def _internal_space(self) -> npt.NDArray[ManimFloat]:
        """
        This is a readonly property which is a custom representation for color space operations.
        It is used for operators and can be used when implementing a custom color space.
        """
        return self._internal_value

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

    @classmethod
    def _construct_from_space(
        cls,
        _space: npt.NDArray[ManimFloat]
        | tuple[float, float, float]
        | tuple[float, float, float, float],
    ) -> Self:
        """
        This function is used as a proxy for constructing a color with an internal value,
        this can be used by subclasses to hook into the construction of new objects using the internal value format
        """
        return cls(_space)

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
    def _internal_from_hex_string(hex_: str, alpha: float) -> ManimColorInternal:
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
            alpha value used for the color if the color is only 3 bytes long, if the color is 4 bytes long the parameter will not be used

        Returns
        -------
        ManimColorInternal
            Internal color representation
        """
        if len(hex_) == 6:
            hex_ += "FF"
        elif len(hex_) == 8:
            alpha = (int(hex_, 16) & 0xFF) / 255
        else:
            raise ValueError(
                "Hex colors must be specified with either 0x or # as prefix and contain 6 or 8 hexadecimal numbers"
            )
        tmp = int(hex_, 16)
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
    def _internal_from_string(name: str, alpha: float) -> ManimColorInternal:
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

        if tmp := _all_color_dict.get(name.upper()):
            tmp._internal_value[3] = alpha
            return tmp._internal_value.copy()
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
        tmp = (self._internal_value[:3] * 255).astype(dtype=np.byte).tobytes()
        return int.from_bytes(tmp, "big")

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
        return np.array(colorsys.rgb_to_hsv(*self.to_rgb()))

    def to_hsl(self) -> HSL_Array_Float:
        """Converts the Manim Color to HSL array.

        .. note::
           Be careful this returns an array in the form `[h, s, l]` where the elements are floats.
           This might be confusing because rgb can also be an array of floats so you might want to annotate the usage
           of this function in your code by typing the variables with :class:`HSL_Array_Float` in order to differentiate
           between rgb arrays and hsl arrays

        Returns
        -------
        HSL_Array_Float
            A hsl array containing 3 elements of type float ranging from 0 to 1
        """
        return np.array(colorsys.rgb_to_hls(*self.to_rgb()))

    def invert(self, with_alpha: bool = False) -> Self:
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
        if with_alpha:
            return self._construct_from_space(1.0 - self._internal_space)
        else:
            alpha = self._internal_space[3]
            new = 1.0 - self._internal_space
            new[-1] = alpha
            return self._construct_from_space(new)

    def interpolate(self, other: Self, alpha: float) -> Self:
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
        return self._construct_from_space(
            self._internal_space * (1 - alpha) + other._internal_space * alpha
        )

    def darker(self, blend: float = 0.2) -> Self:
        """Returns a new color that is darker than the current color, i.e.
        interpolated with black. The opacity is unchanged.

        Parameters
        ----------
        blend : float, optional
            The blend ratio for the interpolation, from 0 (the current color
            unchanged) to 1 (pure black). By default 0.2 which results in a
            slightly darker color

        Returns
        -------
        ManimColor
            The darker ManimColor

        See Also
        --------
        :meth:`lighter`
        """
        from manim.utils.color.manim_colors import BLACK

        alpha = self._internal_space[3]
        black = self._from_internal(BLACK._internal_value)
        return self.interpolate(black, blend).opacity(alpha)

    def lighter(self, blend: float = 0.2) -> Self:
        """Returns a new color that is lighter than the current color, i.e.
        interpolated with white. The opacity is unchanged.

        Parameters
        ----------
        blend : float, optional
            The blend ratio for the interpolation, from 0 (the current color
            unchanged) to 1 (pure white). By default 0.2 which results in a
            slightly lighter color

        Returns
        -------
        ManimColor
            The lighter ManimColor

        See Also
        --------
        :meth:`darker`
        """
        from manim.utils.color.manim_colors import WHITE

        alpha = self._internal_space[3]
        white = self._from_internal(WHITE._internal_value)
        return self.interpolate(white, blend).opacity(alpha)

    def contrasting(
        self,
        threshold: float = 0.5,
        light: Self | None = None,
        dark: Self | None = None,
    ) -> Self:
        """Returns one of two colors, light or dark (by default white or black),
        that contrasts with the current color (depending on its luminance).
        This is typically used to set text in a contrasting color that ensures
        it is readable against a background of the current color.

        Parameters
        ----------
        threshold : float, optional
            The luminance threshold that dictates whether the current color is
            considered light or dark (and thus whether to return the dark or
            light color, respectively), by default 0.5
        light : ManimColor, optional
            The light color to return if the current color is considered dark,
            by default pure white
        dark : ManimColor, optional
            The dark color to return if the current color is considered light,
            by default pure black

        Returns
        -------
        ManimColor
            The contrasting ManimColor
        """
        from manim.utils.color.manim_colors import BLACK, WHITE

        luminance, _, _ = colorsys.rgb_to_yiq(*self.to_rgb())
        if luminance < threshold:
            if light is not None:
                return light
            return self._from_internal(WHITE._internal_value)
        else:
            if dark is not None:
                return dark
            return self._from_internal(BLACK._internal_value)

    def opacity(self, opacity: float) -> Self:
        """Creates a new ManimColor with the given opacity and the same color value as before

        Parameters
        ----------
        opacity : float
            The new opacity value to be used

        Returns
        -------
        ManimColor
            The new ManimColor with the same color value but the new opacity
        """
        tmp = self._internal_space.copy()
        tmp[-1] = opacity
        return self._construct_from_space(tmp)

    def into(self, classtype: type[ManimColorT]) -> ManimColorT:
        """Converts the current color into a different colorspace that is given without changing the _internal_value

        Parameters
        ----------
        classtype : type[ManimColorT]
            The class that is used for conversion, it must be a subclass of ManimColor which respects the specification
            HSV, RGBA, ...

        Returns
        -------
        ManimColorT
            Color object of the type passed into classtype with the same internal value as previously
        """
        return classtype._from_internal(self._internal_value)

    @classmethod
    def _from_internal(cls, value: ManimColorInternal) -> Self:
        """This function is intended to be overwritten by custom color space classes which are subtypes of ManimColor.

        The function constructs a new object of the given class by transforming the value in the internal format ``[r,g,b,a]``
        into a format which the constructor of the custom class can understand. Look at :class:`.HSV` for an example.
        """
        return cls(value)

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
        return cls._from_internal(ManimColor(rgb, alpha)._internal_value)

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
    def from_hex(cls, hex_str: str, alpha: float = 1.0) -> Self:
        """Creates a Manim Color from a hex string, prefixes allowed # and 0x

        Parameters
        ----------
        hex_str : str
            The hex string to be converted (currently only supports 6 nibbles)
        alpha : float, optional
            alpha value to be used for the hex string, by default 1.0

        Returns
        -------
        ManimColor
            The ManimColor represented by the hex string
        """
        return cls._from_internal(ManimColor(hex_str, alpha)._internal_value)

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
        return cls._from_internal(ManimColor(rgb, alpha)._internal_value)

    @classmethod
    def from_hsl(
        cls, hsl: HSL_Array_Float | HSL_Tuple_Float, alpha: float = 1.0
    ) -> Self:
        """Creates a ManimColor from an HSL Array

        Parameters
        ----------
        hsl : HSL_Array_Float | HSL_Tuple_Float
            Any 3 Element Iterable containing floats from 0-1
        alpha : float, optional
            the alpha value to be used, by default 1.0

        Returns
        -------
        ManimColor
            The ManimColor with the corresponding RGB values to the HSL
        """
        rgb = colorsys.hls_to_rgb(*hsl)
        return cls._from_internal(ManimColor(rgb, alpha)._internal_value)

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
        color: ParsableManimColor | Sequence[ParsableManimColor] | None,
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

        def is_sequence(colors: Any) -> TypeGuard[Sequence[ParsableManimColor]]:
            return isinstance(colors, (list, tuple))

        def is_parsable(color: Any) -> TypeGuard[ParsableManimColor]:
            return not isinstance(color, (list, tuple))

        if is_sequence(color):
            return [
                cls._from_internal(ManimColor(c, alpha)._internal_value) for c in color
            ]
        elif is_parsable(color):
            return cls._from_internal(ManimColor(color, alpha)._internal_value)
        else:
            return cls._from_internal(ManimColor("WHITE", alpha)._internal_value)

    @staticmethod
    def gradient(colors: list[ManimColor], length: int) -> None:
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
        value: bool = np.allclose(self._internal_value, other._internal_value)
        return value

    def __add__(self, other: int | float | Self) -> Self:
        if isinstance(other, (int, float)):
            return self._construct_from_space(self._internal_space + other)
        else:
            return self._construct_from_space(
                self._internal_space + other._internal_space
            )

    def __radd__(self, other: int | float | Self) -> Self:
        return self + other

    def __sub__(self, other: int | float | Self) -> Self:
        if isinstance(other, (int, float)):
            return self._construct_from_space(self._internal_space - other)
        else:
            return self._construct_from_space(
                self._internal_space - other._internal_space
            )

    def __rsub__(self, other: int | float | Self) -> Self:
        return self - other

    def __mul__(self, other: int | float | Self) -> Self:
        if isinstance(other, (int, float)):
            return self._construct_from_space(self._internal_space * other)
        else:
            return self._construct_from_space(
                self._internal_space * other._internal_space
            )

    def __rmul__(self, other: int | float | Self) -> Self:
        return self * other

    def __truediv__(self, other: int | float | Self) -> Self:
        if isinstance(other, (int, float)):
            return self._construct_from_space(self._internal_space / other)
        else:
            return self._construct_from_space(
                self._internal_space / other._internal_space
            )

    def __rtruediv__(self, other: int | float | Self) -> Self:
        return self / other

    def __floordiv__(self, other: int | float | Self) -> Self:
        if isinstance(other, (int, float)):
            return self._construct_from_space(self._internal_space // other)
        else:
            return self._construct_from_space(
                self._internal_space // other._internal_space
            )

    def __rfloordiv__(self, other: int | float | Self) -> Self:
        return self // other

    def __mod__(self, other: int | float | Self) -> Self:
        if isinstance(other, (int, float)):
            return self._construct_from_space(self._internal_space % other)
        else:
            return self._construct_from_space(
                self._internal_space % other._internal_space
            )

    def __rmod__(self, other: int | float | Self) -> Self:
        return self % other

    def __pow__(self, other: int | float | Self) -> Self:
        if isinstance(other, (int, float)):
            return self._construct_from_space(self._internal_space**other)
        else:
            return self._construct_from_space(
                self._internal_space**other._internal_space
            )

    def __rpow__(self, other: int | float | Self) -> Self:
        return self**other

    def __invert__(self) -> Self:
        return self.invert()

    def __int__(self) -> int:
        return self.to_integer()

    def __getitem__(self, index: int) -> float:
        value: float = self._internal_space[index]
        return value

    def __and__(self, other: Self) -> Self:
        return self._construct_from_space(
            self._internal_from_integer(self.to_integer() & int(other), 1.0)
        )

    def __or__(self, other: Self) -> Self:
        return self._construct_from_space(
            self._internal_from_integer(self.to_integer() | int(other), 1.0)
        )

    def __xor__(self, other: Self) -> Self:
        return self._construct_from_space(
            self._internal_from_integer(self.to_integer() ^ int(other), 1.0)
        )


RGBA = ManimColor
"""RGBA Color Space"""


class HSV(ManimColor):
    """HSV Color Space"""

    def __init__(
        self,
        hsv: HSV_Array_Float | HSV_Tuple_Float | HSVA_Array_Float | HSVA_Tuple_Float,
        alpha: float = 1.0,
    ) -> None:
        super().__init__(None)
        if len(hsv) == 3:
            self.__hsv: HSVA_Array_Float = np.asarray((*hsv, alpha))
        elif len(hsv) == 4:
            self.__hsv = np.asarray(hsv)
        else:
            raise ValueError("HSV Color must be an array of 3 values")

    @classmethod
    @override
    def _from_internal(cls, value: ManimColorInternal) -> Self:
        hsv = colorsys.rgb_to_hsv(*value[:3])
        hsva = [*hsv, value[-1]]
        return cls(np.array(hsva))

    @property
    def hue(self) -> float:
        value: float = self.__hsv[0]
        return value

    @hue.setter
    def hue(self, value: float) -> None:
        self.__hsv[0] = value

    @property
    def saturation(self) -> float:
        value: float = self.__hsv[1]
        return value

    @saturation.setter
    def saturation(self, value: float) -> None:
        self.__hsv[1] = value

    @property
    def value(self) -> float:
        value: float = self.__hsv[2]
        return value

    @value.setter
    def value(self, value: float) -> None:
        self.__hsv[2] = value

    @property
    def h(self) -> float:
        value: float = self.__hsv[0]
        return value

    @h.setter
    def h(self, value: float) -> None:
        self.__hsv[0] = value

    @property
    def s(self) -> float:
        value: float = self.__hsv[1]
        return value

    @s.setter
    def s(self, value: float) -> None:
        self.__hsv[1] = value

    @property
    def v(self) -> float:
        value: float = self.__hsv[2]
        return value

    @v.setter
    def v(self, value: float) -> None:
        self.__hsv[2] = value

    @property
    def _internal_space(self) -> npt.NDArray:
        return self.__hsv

    @property
    def _internal_value(self) -> ManimColorInternal:
        """Returns the internal value of the current Manim color [r,g,b,a] float array

        Returns
        -------
        ManimColorInternal
            internal color representation
        """
        return np.array(
            [
                *colorsys.hsv_to_rgb(self.__hsv[0], self.__hsv[1], self.__hsv[2]),
                self.__alpha,
            ],
            dtype=ManimColorDType,
        )

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
        tmp = colorsys.rgb_to_hsv(value[0], value[1], value[2])
        self.__hsv = np.array(tmp)
        self.__alpha = value[3]


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
    color1: ManimColorT, color2: ManimColorT, alpha: float
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
    "HSV",
    "RGBA",
]
