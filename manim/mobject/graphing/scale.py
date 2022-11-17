from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

__all__ = ["LogBase", "LinearBase"]

from manim.mobject.text.numbers import Integer

if TYPE_CHECKING:
    from manim.mobject.mobject import Mobject


class _ScaleBase:
    """Scale baseclass for graphing/functions."""

    def __init__(self, custom_labels: bool = False):
        """
        Parameters
        ----------
        custom_labels
            Whether to create custom labels when plotted on a :class:`~.NumberLine`.
        """
        self.custom_labels = custom_labels

    def function(self, value: float) -> float:
        """The function that will be used to scale the values.

        Parameters
        ----------
        value
            The number/``np.ndarray`` to be scaled.

        Returns
        -------
        float
            The value after it has undergone the scaling.

        Raises
        ------
        NotImplementedError
            Must be subclassed.
        """
        raise NotImplementedError

    def inverse_function(self, value: float) -> float:
        """The inverse of ``function``. Used for plotting on a particular axis.

        Raises
        ------
        NotImplementedError
            Must be subclassed.
        """
        raise NotImplementedError

    def get_custom_labels(
        self,
        val_range: Iterable[float],
    ) -> Iterable[Mobject]:
        """Custom instructions for generating labels along an axis.

        Parameters
        ----------
        val_range
            The position of labels. Also used for defining the content of the labels.

        Returns
        -------
        Dict
            A list consisting of the labels.
            Can be passed to :meth:`~.NumberLine.add_labels() along with ``val_range``.

        Raises
        ------
        NotImplementedError
            Can be subclassed, optional.
        """
        raise NotImplementedError


class LinearBase(_ScaleBase):
    def __init__(self, scale_factor: float = 1.0):
        """The default scaling class.

        Parameters
        ----------
        scale_factor
            The slope of the linear function, by default 1.0
        """

        super().__init__()
        self.scale_factor = scale_factor

    def function(self, value: float) -> float:
        """Multiplies the value by the scale factor.

        Parameters
        ----------
        value
            Value to be multiplied by the scale factor.
        """
        return self.scale_factor * value

    def inverse_function(self, value: float) -> float:
        """Inverse of function. Divides the value by the scale factor.

        Parameters
        ----------
        value
            value to be divided by the scale factor.
        """
        return value / self.scale_factor


class LogBase(_ScaleBase):
    def __init__(self, base: float = 10, custom_labels: bool = True):
        """Scale for logarithmic graphs/functions.

        Parameters
        ----------
        base
            The base of the log, by default 10.
        custom_labels
            For use with :class:`~.Axes`:
            Whetherer or not to include ``LaTeX`` axis labels, by default True.

        Examples
        --------
        .. code-block:: python

            func = ParametricFunction(lambda x: x, scaling=LogBase(base=2))

        """
        super().__init__()
        self.base = base
        self.custom_labels = custom_labels

    def function(self, value: float) -> float:
        """Scales the value to fit it to a logarithmic scale.``self.function(5)==10**5``"""
        return self.base**value

    def inverse_function(self, value: float) -> float:
        """Inverse of ``function``. The value must be greater than 0"""
        if isinstance(value, np.ndarray):
            condition = value.any() <= 0
            func = lambda value, base: np.log(value) / np.log(base)
        else:
            condition = value <= 0
            func = math.log

        if condition:
            raise ValueError(
                "log(0) is undefined. Make sure the value is in the domain of the function"
            )
        value = func(value, self.base)
        return value

    def get_custom_labels(
        self,
        val_range: Iterable[float],
        unit_decimal_places: int = 0,
        **base_config: dict[str, Any],
    ) -> list[Mobject]:
        """Produces custom :class:`~.Integer` labels in the form of ``10^2``.

        Parameters
        ----------
        val_range
            The iterable of values used to create the labels. Determines the exponent.
        unit_decimal_places
            The number of decimal places to include in the exponent
        base_config
            Additional arguments to be passed to :class:`~.Integer`.
        """

        # uses `format` syntax to control the number of decimal places.
        tex_labels = [
            Integer(
                self.base,
                unit="^{%s}" % (f"{self.inverse_function(i):.{unit_decimal_places}f}"),
                **base_config,
            )
            for i in val_range
        ]
        return tex_labels
