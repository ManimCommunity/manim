import copy
import operator
from functools import wraps

import numpy as np

from ..config import config as global_config


def percent_to_munit(val, dim, config):
    return val / 100 * _size_from_dimension(dim, config=config)


def pixels_to_munit(val, dim, config):
    return (
        val
        / _size_from_dimension(dim, True, config=config)
        * _size_from_dimension(dim, config=config)
    )


def _size_from_dimension(
    dim=0, pixels=False, config=global_config, treat_dim2_as_dim0=True
):
    if dim == 0 or (dim == 2 and treat_dim2_as_dim0):
        return config["frame_width"] if not pixels else config["pixel_width"]
    elif dim == 1:
        return config["frame_height"] if not pixels else config["pixel_height"]
    else:
        # TODO: determine size in z-direction
        raise ValueError(
            f"Z-direction is not supported for units - set treat_dim2_as_dim0 or use do not use z-dim in a unit."
        )


def _is_dimensional(value):
    return isinstance(value, np.ndarray) and value.shape[0] <= 3


class Unit:
    def __init__(self, value):
        self._value = value
        self._converter = lambda val, dim, config: val

    @property
    def value(self):
        return copy.deepcopy(self._value)

    @property
    def converter(self):
        return self._converter

    def convert(self, dim=None, config=global_config):
        if dim:
            return self.converter(self.value, dim, config)
        elif _is_dimensional(self.value):
            return_value = self.value

            for i in range(0, return_value.shape[0]):
                return_value[i] = self.converter(return_value[i], i, config)
            return return_value
        else:
            raise ValueError("Unable to determine dimension.")

    def apply_operator(self, other, op):
        if issubclass(type(other), Unit):
            if isinstance(other, type(self)):
                # Create and return a new instance
                return type(self)(op(self.value, other.value))
            else:
                # For now lets not bother with operators on different units
                raise NotImplementedError
        # Create and return a new instance
        return type(self)(op(self.value, other))

    def __add__(self, other):
        return self.apply_operator(other, operator.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __floordiv__(self, other):
        return self.apply_operator(other, operator.floordiv)

    def __rfloordiv__(self, other):
        return self.__floordiv__(other)

    def __mul__(self, other):
        return self.apply_operator(other, operator.mul)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self.apply_operator(other, operator.sub)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __truediv__(self, other):
        return self.apply_operator(other, operator.truediv)

    def __rtruediv__(self, other):
        return self.__truediv__(other)


class MUnit(Unit):
    def __init__(self, value):
        super(MUnit, self).__init__(value)


class PixelUnit(Unit):
    def __init__(self, value):
        super(PixelUnit, self).__init__(value)
        self._converter = pixels_to_munit


class PercentUnit(Unit):
    def __init__(self, value):
        super(PercentUnit, self).__init__(value)
        self._converter = percent_to_munit


def handle_units(f, dim=None, config=global_config):
    @wraps(f)
    def wrapper(*args, **kwargs):
        # TODO figure what to do when dim=None, preferable determine the dim.
        return f(
            *[v.convert(dim, config) if isinstance(v, Unit) else v for v in args],
            **{
                k: v.convert(dim, config) if isinstance(v, Unit) else v
                for k, v in kwargs.items()
            },
        )

    return wrapper
