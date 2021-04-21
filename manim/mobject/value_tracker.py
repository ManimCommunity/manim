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

                pointer_value = ValueTracker(0)
                pointer.add_updater(
                    lambda m: m.next_to(
                                number_line.n2p(pointer_value.get_value()),
                                UP
                            )
                )
                self.add(number_line, pointer,label)
                pointer_value += 1.5
                self.wait(1)
                pointer_value -= 4
                self.wait(0.5)
                self.play(pointer_value.animate.set_value(5)),
                self.wait(0.5)
                self.play(pointer_value.animate.set_value(3))
                self.play(pointer_value.animate.increment_value(-2))
                self.wait(0.5)
    """

    def __init__(self, value=0, **kwargs):
        Mobject.__init__(self, **kwargs)
        self.points = np.zeros((1, 3))
        self.set_value(value)

    def get_value(self) -> float:
        """Get the current value of the ValueTracker. This value changes continuously
        when :attr:`animate` for the ValueTracker is called."""
        return self.points[0, 0]

    def set_value(self, value: Union[float, int]):
        """Sets a new scalar value to the ValueTracker"""
        self.points[0, 0] = value
        return self

    def increment_value(self, d_value: Union[float, int]):
        """Increments (adds) a scalar value  to the ValueTracker"""
        self.set_value(self.get_value() + d_value)
    
    def __bool__(self):
        """Allows ValueTracker to be tested directly in boolean expressions. True if the value of the ValueTracker is nonzero."""
        return bool(self.get_value())

    def __iadd__(self, d_value: Union[float, int]):
        """adds ``+=`` syntax to increment the value of the ValueTracker"""
        self.increment_value(d_value)
        return self
    
    def __ifloordiv__(self, d_value: Union[float, int]):
        """adds ``//=`` syntax to floor divide the value of the ValueTracker"""
        self.set_value(self.get_value() // d_value)

    def __imod__(self, d_value: Union[float, int]):
        """adds ``%=`` syntax to floor mod the value of the ValueTracker"""
        self.set_value(self.get_value() % d_value)
    
    def __imul__(self, d_value: Union[float, int]):
        """adds ``*=`` syntax to multiply the value of the ValueTracker"""
        self.set_value(self.get_value() * d_value)
    
    def __ipow__(self, d_value: Union[float, int]):
        """adds ``**=`` syntax to exponentiate the value of the ValueTracker"""
    
    def __isub__(self, d_value: Union[float, int]):
        """adds ``-=`` syntax to decrement the value of the ValueTracker"""
        self.increment_value(-d_value)
        return self
    
    def __itruediv__(self, d_value: Union[float, int]):
        """adds ``/=`` syntax to floor divide the value of the ValueTracker"""
        self.set_value(self.get_value() / d_value)

    def interpolate(self, mobject1, mobject2, alpha, path_func=straight_path):
        """
        Turns self into an interpolation between mobject1
        and mobject2.
        """
        self.points = path_func(mobject1.points, mobject2.points, alpha)
        return self


class ExponentialValueTracker(ValueTracker):
    """Operates just like ValueTracker, except it encodes the value as the
    exponential of a position coordinate, which changes how interpolation
    behaves.

    Note that ExponentialValueTracker does not handle non-positive values.

    Examples
    --------
    .. manim:: ExponentialValueTrackerExample

        class ExponentialValueTrackerExample(Scene):
            def construct(self):
                number_line = NumberLine()
                pointer = Vector(DOWN)
                label = MathTex("x").add_updater(lambda m: m.next_to(pointer, UP))

                pointer_value = ExponentialValueTracker(4)
                pointer.add_updater(
                    lambda m: m.next_to(
                                pointer_value.get_value() * RIGHT,
                                UP
                            )
                )
                self.add(number_line, pointer,label)

                self.play(pointer_value.animate.set_value(0.5))
                self.wait(0.5)
                self.play(pointer_value.animate.set_value(6))
                self.wait(0.5)
                self.play(pointer_value.animate.set_value(3))
                self.wait(0.5)
                self.play(pointer_value.animate.set_value(2))
                self.wait(0.5)
    """

    def get_value(self):
        """Get the current value of the ExponentialValueTracker."""
        return np.exp(ValueTracker.get_value(self))

    def set_value(self, value):
        """Set a new scalar value to the ExponentialValueTracker. The value cannot
        be non-positive."""
        return ValueTracker.set_value(self, np.log(value))


class ComplexValueTracker(ValueTracker):
    """Operates like ValueTracker, except it encodes a complex-valued
    parameter as opposed to a real-valued parameter.

    Examples
    --------
    .. manim:: ComplexValueTrackerExample

        class ComplexValueTrackerExample(Scene):
            def construct(self):
                tracker = ComplexValueTracker(-2+1j)
                dot = Dot().add_updater(
                    lambda x: x.move_to(tracker.points)
                )

                self.add(NumberPlane(), dot)

                self.play(tracker.animate.set_value(3+2j))
                self.play(tracker.animate.set_value(tracker.get_value() * 1j))
                self.play(tracker.animate.set_value(tracker.get_value() - 2j))
                self.play(tracker.animate.set_value(tracker.get_value() / (-2 + 3j)))
    """

    def get_value(self):
        """Get the current value of the ComplexValueTracker. This value changes
        continuously when :attr:`animate` for the ComplexValueTracker is called."""
        return complex(*self.points[0, :2])

    def set_value(self, z):
        """Sets a new complex value to the ComplexValueTracker

        When the value is set through :attr:`animate`, the value will take a straight
        path from the source point to the destination point."""
        z = complex(z)
        self.points[0, :2] = (z.real, z.imag)
        return self
