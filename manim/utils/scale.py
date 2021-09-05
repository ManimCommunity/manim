import math
from typing import TYPE_CHECKING, Any, Dict, Iterable

__all__ = ["LogBase", "LinearBase"]

from ..mobject.numbers import Integer

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
    ) -> Dict[Iterable[float], Iterable["Mobject"]]:
        """Custom instructions for generating labels along an axis.

        Parameters
        ----------
        val_range
            The position of labels. Also used for defining the content of the labels.

        Returns
        -------
        Dict
            A dict consistiong of the position/labels. Passed to :meth:`~.NumberLine.add_labels()`

        Raises
        ------
        NotImplementedError
            Can be subclassed, optional.
        """
        raise NotImplementedError


class LinearBase(_ScaleBase):
    def __init__(self, scale_factor: float = 1.0):
        super().__init__()
        self.scale_factor = scale_factor

    def function(self, value):
        return self.scale_factor * value

    def inverse_function(self, value):
        return value / self.scale_factor


class LogBase(_ScaleBase):
    def __init__(self, base: float = 10, custom_labels: bool = True):
        """Scale for logarithmic graphs/functions.

        Parameters
        ----------
        base
            The base of the log, by default 10.
        custom_labels : bool, optional
            Whetherer or not to include ``LaTeX`` axis labels, by default True.
        """
        super().__init__()
        self.base = base
        self.custom_labels = custom_labels

    def function(self, value: float) -> float:
        """Scales the value to fit it to a logarithmic scale.``self.function(5) == 10**5``"""
        return self.base ** value

    def inverse_function(self, value: float) -> float:
        """Inverse of ``function``."""
        value = math.log(value, self.base)
        return value

    def get_custom_labels(
        self,
        val_range: Iterable[float],
        unit_decimal_places: int = 0,
        **base_config: Dict[str, Any],
    ) -> Dict:
        # uses `format` syntax to control the number of decimal places.
        tex_labels = [
            Integer(
                self.base,
                unit="^{%s}" % (f"{self.inverse_function(i):.{unit_decimal_places}f}"),
                **base_config,
            )
            for i in val_range
        ]
        labels = dict(zip(val_range, tex_labels))
        return labels
