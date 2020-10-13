import copy
from functools import wraps

import numpy as np

from ..config import config as global_config


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


class MUnit(Unit):
    def __init__(self, value):
        super(MUnit, self).__init__(value)


class PixelUnit(Unit):
    def __init__(self, value):
        super(PixelUnit, self).__init__(value)
        self._converter = (
            lambda val, dim, config: val
            / _size_from_dimension(dim, True, config)
            * _size_from_dimension(dim, config=config)
        )


class PercentUnit(Unit):
    def __init__(self, value):
        super(PercentUnit, self).__init__(value)
        self._converter = (
            lambda val, dim, config: val / 100 * _size_from_dimension(dim, config)
        )


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
