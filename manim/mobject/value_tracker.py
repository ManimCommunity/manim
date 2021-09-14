"""Mobjects that dynamically show the change of a variable."""

__all__ = ["ValueTracker", "ComplexValueTracker"]


import numpy as np

from ..mobject.mobject import Mobject
from ..utils.paths import straight_path
from .opengl_compatibility import ConvertToOpenGL


class ValueTracker(Mobject, metaclass=ConvertToOpenGL):
    """A mobject that can be used for tracking (real-valued) parameters.
    Useful for animating parameter changes.

    Not meant to be displayed.  Instead the position encodes some
    number, often one which another animation or continual_animation
    uses for its update function, and by treating it as a mobject it can
    still be animated and manipulated just like anything else.

    This value changes continuously when animated using the :attr:`animate` syntax.

    Examples
    --------
    .. manim:: ValueTrackerExample

        class ValueTrackerExample(Scene):
            def construct(self):
                number_line = NumberLine()
                pointer = Vector(DOWN)
                label = MathTex("x").add_updater(lambda m: m.next_to(pointer, UP))

                tracker = ValueTracker(0)
                pointer.add_updater(
                    lambda m: m.next_to(
                                number_line.n2p(tracker.get_value()),
                                UP
                            )
                )
                self.add(number_line, pointer,label)
                tracker += 1.5
                self.wait(1)
                tracker -= 4
                self.wait(0.5)
                self.play(tracker.animate.set_value(5)),
                self.wait(0.5)
                self.play(tracker.animate.set_value(3))
                self.play(tracker.animate.increment_value(-2))
                self.wait(0.5)

    .. note::

        You can also link ValueTrackers to updaters. In this case, you have to make sure that the
        ValueTracker is added to the scene by ``add``

    .. manim:: ValueTrackerExample

        class ValueTrackerExample(Scene):
            def construct(self):
                tracker = ValueTracker(0)
                label = Dot(radius=3).add_updater(lambda x : x.set_x(tracker.get_value()))
                self.add(label)
                self.add(tracker)
                tracker.add_updater(lambda mobject, dt: mobject.increment_value(dt))
                self.wait(2)

    """

    def __init__(self, value=0, **kwargs):
        super().__init__(**kwargs)
        self.set_points(np.zeros((1, 3)))
        self.set_value(value)

    def get_value(self) -> float:
        """Get the current value of this ValueTracker."""
        return self.points[0, 0]

    def set_value(self, value: float):
        """Sets a new scalar value to the ValueTracker"""
        self.points[0, 0] = value
        return self

    def increment_value(self, d_value: float):
        """Increments (adds) a scalar value  to the ValueTracker"""
        self.set_value(self.get_value() + d_value)
        return self

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
        self.set_points(path_func(mobject1.points, mobject2.points, alpha))
        return self


class ComplexValueTracker(ValueTracker):
    """Tracks a complex-valued parameter.

    When the value is set through :attr:`animate`, the value will take a straight path from the
    source point to the destination point.

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
        """Get the current value of this value tracker as a complex number.

        The value is internally stored as a points array [a, b, 0]. This can be accessed directly
        to represent the value geometrically, see the usage example."""
        return complex(*self.points[0, :2])

    def set_value(self, z):
        """Sets a new complex value to the ComplexValueTracker"""
        z = complex(z)
        self.points[0, :2] = (z.real, z.imag)
        return self
