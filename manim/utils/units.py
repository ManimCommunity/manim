from functools import wraps


class Unit():
    def __init__(self, value, unit):
        self._value = value
        self._unit = unit

    @property
    def value(self):
        return self._value


def accepts_unit(f):
    @wraps(f)
    def wrapper(*args):
        return f(*[a.value if isinstance(a, Unit) else a for a in args])

    return wrapper
