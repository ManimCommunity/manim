from functools import wraps

from ..config import config

PERCENT = "%"
PIXELS = "px"
MUNIT = "m"


def _size_from_dimension(dim=0, pixels=False):
    if dim == 0:
        return config["frame_width"] if not pixels else config["pixel_width"]
    elif dim == 1:
        return config["frame_height"] if not pixels else config["pixel_height"]
    else:
        # TODO: determine size in z-direction
        raise ValueError("Dimension not supported.")


class Unit():
    def __init__(self, value, unit=None):
        self._value = value
        self._unit = unit

    def value(self, dim=0):
        if not self._unit or self._unit == MUNIT:
            return self._value
        elif self._unit == PIXELS:
            return self._value / _size_from_dimension(dim, True) * _size_from_dimension(dim)
        elif self._unit == PERCENT:
            return self._value * _size_from_dimension(dim)
        else:
            raise ValueError("Unsupported unit.")


def accepts_unit(f):
    @wraps(f)
    def wrapper(*args):
        return f(*[a.value if isinstance(a, Unit) else a for a in args])

    return wrapper
