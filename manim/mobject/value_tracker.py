import numpy as np
import attr
import typing as tp

from ..utils.paths import straight_path
from ..mobject.mobject import Mobject


@attr.s(auto_attribs=True, eq=False)
class ValueTracker(Mobject):
    """
    Not meant to be displayed.  Instead the position encodes some
    number, often one which another animation or continual_animation
    uses for its update function, and by treating it as a mobject it can
    still be animated and manipulated just like anything else.
    """
    value: tp.Any = 0

    def __attrs_post_init__(self):
        Mobject.__attrs_post_init__(self)
        self.points = np.zeros((1, 3))
        self.set_value(self.value)

    def get_value(self):
        return self.points[0, 0]

    def set_value(self, value):
        self.points[0, 0] = value
        return self

    def increment_value(self, d_value):
        self.set_value(self.get_value() + d_value)

    def interpolate(self, mobject1, mobject2, alpha, path_func=straight_path):
        """
        Turns self into an interpolation between mobject1
        and mobject2.
        """
        self.points = path_func(mobject1.points, mobject2.points, alpha)
        return self


@attr.s(auto_attribs=True, eq=False)
class ExponentialValueTracker(ValueTracker):
    """
    Operates just like ValueTracker, except it encodes the value as the
    exponential of a position coordinate, which changes how interpolation
    behaves
    """
    def __attrs_post_init__(self):
        ValueTracker.__attrs_post_init__(self)

    def get_value(self):
        return np.exp(ValueTracker.get_value(self))

    def set_value(self, value):
        return ValueTracker.set_value(self, np.log(value))


@attr.s(auto_attribs=True, eq=False)
class ComplexValueTracker(ValueTracker):
    def __attrs_post_init__(self):
        ValueTracker.__attrs_post_init__(self)

    def get_value(self):
        return complex(*self.points[0, :2])

    def set_value(self, z):
        z = complex(z)
        self.points[0, :2] = (z.real, z.imag)
        return self
