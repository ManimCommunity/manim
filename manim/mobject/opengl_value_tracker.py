"""Mobjects that dynamically show the change of a variable."""

__all__ = [
    "OpenGLValueTracker",
    "OpenGLComplexValueTracker",
]


import numpy as np

from ..mobject.opengl_mobject import OpenGLMobject
from ..utils.paths import straight_path


class OpenGLValueTracker(OpenGLMobject):
    """A mobject that can be used for tracking (real-valued) parameters.
    Useful for animating parameter changes.
    Not meant to be displayed.  Instead the position encodes some
    number, often one which another animation or continual_animation
    uses for its update function, and by treating it as a mobject it can
    still be animated and manipulated just like anything else.

    .. note::
        You can also link ValueTrackers to updaters. In this case, you have to make sure that the ValueTracker is added to the scene by ``add``
    """

    def __init__(self, value=0, **kwargs):
        OpenGLMobject.__init__(self, **kwargs)
        self.points = np.zeros((1, 3))
        self.set_value(value)

    def get_value(self) -> float:
        """Get the current value of the ValueTracker. This value changes continuously when :attr:`animate` for the ValueTracker is called."""
        return self.points[0, 0]

    def set_value(self, value: float):
        """Sets a new scalar value to the ValueTracker"""
        self.points[0, 0] = value
        return self

    def increment_value(self, d_value: float):
        """Increments (adds) a scalar value  to the ValueTracker"""
        self.set_value(self.get_value() + d_value)

    def __bool__(self):
        """Return whether the value of this value tracker evaluates as true."""
        return bool(self.get_value())

    def __iadd__(self, d_value: float):
        """adds ``+=`` syntax to increment the value of the ValueTracker"""
        self.increment_value(d_value)
        return self

    def __ifloordiv__(self, d_value: float):
        """Set the value of this value tracker to the floor division of the current value by ``d_value``."""
        self.set_value(self.get_value() // d_value)
        return self

    def __imod__(self, d_value: float):
        """Set the value of this value tracker to the current value modulo ``d_value``."""
        self.set_value(self.get_value() % d_value)
        return self

    def __imul__(self, d_value: float):
        """Set the value of this value tracker to the product of the current value and ``d_value``."""
        self.set_value(self.get_value() * d_value)
        return self

    def __ipow__(self, d_value: float):
        """Set the value of this value tracker to the current value raised to the power of ``d_value``."""
        self.set_value(self.get_value() ** d_value)
        return self

    def __isub__(self, d_value: float):
        """adds ``-=`` syntax to decrement the value of the ValueTracker"""
        self.increment_value(-d_value)
        return self

    def __itruediv__(self, d_value: float):
        """Sets the value of this value tracker to the current value divided by ``d_value``."""
        self.set_value(self.get_value() / d_value)
        return self

    def interpolate(self, mobject1, mobject2, alpha, path_func=straight_path):
        """
        Turns self into an interpolation between mobject1
        and mobject2.
        """
        self.points = path_func(mobject1.points, mobject2.points, alpha)
        return self


class OpenGLComplexValueTracker(OpenGLValueTracker):
    def get_value(self):
        return complex(*self.points[0, :2])

    def set_value(self, z):
        z = complex(z)
        self.points[0, :2] = (z.real, z.imag)
        return self
