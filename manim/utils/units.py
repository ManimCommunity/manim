import copy
import logging
from functools import wraps

import numpy as np

from ..config import config


def _size_from_dimension(dim=0, pixels=False):
    if dim == 0:
        return config["frame_width"] if not pixels else config["pixel_width"]
    elif dim == 1:
        return config["frame_height"] if not pixels else config["pixel_height"]
    else:
        # TODO: determine size in z-direction
        logging.getLogger("manim").warning(
            f"Z-direction is not supported for units - returning size of x-dimension."
        )
        return config["frame_width"] if not pixels else config["pixel_width"]


def _is_dimensional(value):
    return isinstance(value, np.ndarray) and value.shape[0] <= 3


class Unit:
    def __init__(self, value):
        self._value = value
        self._converter = lambda val, dim: val

    @property
    def value(self):
        return copy.deepcopy(self._value)

    @property
    def converter(self):
        return self._converter

    def convert(self, dim=None):
        if dim:
            return self.converter(self.value, dim=dim)
        elif _is_dimensional(self.value):
            return_value = self.value

            for i in range(0, return_value.shape[0]):
                return_value[i] = self.converter(return_value[i], i)
            return return_value
        else:
            raise ValueError("Unable to determine dimension.")


class MUnit(Unit):
    def __init__(self, value):
        super(MUnit, self).__init__(value)


class PixelUnit(Unit):
    def __init__(self, value):
        super(PixelUnit, self).__init__(value)
        self._converter = (
            lambda val, dim: val
            / _size_from_dimension(dim, True)
            * _size_from_dimension(dim)
        )


class PercentUnit(Unit):
    def __init__(self, value):
        super(PercentUnit, self).__init__(value)
        self._converter = lambda val, dim: val / 100 * _size_from_dimension(dim)


def handle_units(f, dim=None):
    @wraps(f)
    def wrapper(*args, **kwargs):
        # TODO figure what to do when dim=None, preferable determine the dim.
        return f(
            *[v.convert(dim) if isinstance(v, Unit) else v for v in args],
            **{
                k: v.convert(dim) if isinstance(v, Unit) else v
                for k, v in kwargs.items()
            },
        )

    return wrapper
