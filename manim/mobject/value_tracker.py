"""Mobjects that dynamically show the change of a variable."""

__all__ = ["ValueTracker", "ExponentialValueTracker", "ComplexValueTracker"]


from typing import Union

import numpy as np

from ..mobject.mobject import Mobject
from ..utils.paths import straight_path


class ValueTracker(Mobject):
    """A mobject that can be used for tracking (real-valued) parameters.
    Useful for animating parameter changes.

    Not meant to be displayed.  Instead the position encodes some
    number, often one which another animation or continual_animation
    uses for its update function, and by treating it as a mobject it can
    still be animated and manipulated just like anything else.

    Examples
    --------
    .. manim:: ValueTrackerExample

        class ValueTrackerExample(Scene):
            def construct(self):
                number_line = NumberLine()
                pointer = Vector(DOWN)
                label = MathTex("x").add_updater(lambda m: m.next_to(pointer, UP))

                pointer_tracker = ValueTracker(0)
                pointer.add_updater(
                    lambda m: m.next_to(
                                number_line.n2p(pointer_tracker.get_value()),
                                UP
                            )
                )
                self.add(number_line, pointer,label)
                pointer_tracker += 1.5
                self.wait(1)
                pointer_tracker -= 4
                self.wait(0.5)
                self.play(pointer_tracker.animate.set_value(5)),
                self.wait(0.5)
                self.play(pointer_tracker.animate.set_value(3))
                self.play(pointer_tracker.animate.increment_value(-2))
                self.wait(0.5)
    """

    def __init__(self, value=0, **kwargs):
        Mobject.__init__(self, **kwargs)
        self.points = np.zeros((1, 3))
        self.set_value(value)

    def get_value(self) -> float:
        """Get the current value of the ValueTracker. This value changes continuously when :attr:`animate` for the ValueTracker is called."""
        return self.points[0, 0]

    def set_value(self, value: Union[float, int]):
        """Sets a new scalar value to the ValueTracker"""
        self.points[0, 0] = value
        return self

    def increment_value(self, d_value: Union[float, int]):
        """Increments (adds) a scalar value  to the ValueTracker"""
        self.set_value(self.get_value() + d_value)

    def __iadd__(self, d_value: Union[float, int]):
        """adds ``+=`` syntax to increment the value of the ValueTracker"""
        self.increment_value(d_value)
        return self

    def __isub__(self, d_value: Union[float, int]):
        """adds ``-=`` syntax to decrement the value of the ValueTracker"""
        self.increment_value(-d_value)
        return self

    def interpolate(self, mobject1, mobject2, alpha, path_func=straight_path):
        """
        Turns self into an interpolation between mobject1
        and mobject2.
        """
        self.points = path_func(mobject1.points, mobject2.points, alpha)
        return self


class ExponentialValueTracker(ValueTracker):
    """
    Operates just like ValueTracker, except it encodes the value as the
    exponential of a position coordinate, which changes how interpolation
    behaves
    """

    def get_value(self):
        return np.exp(ValueTracker.get_value(self))

    def set_value(self, value):
        return ValueTracker.set_value(self, np.log(value))


class ComplexValueTracker(ValueTracker):
    def get_value(self):
        return complex(*self.points[0, :2])

    def set_value(self, z):
        z = complex(z)
        self.points[0, :2] = (z.real, z.imag)
        return self
