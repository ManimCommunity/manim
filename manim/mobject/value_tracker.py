"""Simple mobjects that can be used for storing (and updating) a value."""

from __future__ import annotations

__all__ = ["ValueTracker", "ComplexValueTracker"]

from typing import TYPE_CHECKING, Any

import numpy as np
from collections.abc import Sequence

from manim.mobject.mobject import Mobject
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.utils.paths import straight_path

if TYPE_CHECKING:
    from typing import Self

    from manim.typing import PathFuncType


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
                self.play(tracker.animate.set_value(5))
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

    def __init__(self, value: float = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.set(points=np.zeros((1, 3)))
        self.set_value(value)

    def get_value(self) -> float:
        """Get the current value of this ValueTracker."""
        value: float = self.points[0, 0]
        return value

    def set_value(self, value: float) -> Self:
        """Sets a new scalar value to the ValueTracker."""
        self.points[0, 0] = value
        return self

    def increment_value(self, d_value: float) -> Self:
        """Increments (adds) a scalar value to the ValueTracker."""
        self.set_value(self.get_value() + d_value)
        return self

    def __bool__(self) -> bool:
        """Return whether the value of this ValueTracker evaluates as true."""
        return bool(self.get_value())

    def __add__(self, d_value: float | Mobject) -> ValueTracker:
        """Return a new :class:`ValueTracker` whose value is the current tracker's value plus
        ``d_value``.
        """
        if isinstance(d_value, Mobject):
            raise ValueError(
                "Cannot increment ValueTracker by a Mobject. Please provide a scalar value."
            )
        return ValueTracker(self.get_value() + d_value)

    def __iadd__(self, d_value: float | Mobject) -> Self:
        """adds ``+=`` syntax to increment the value of the ValueTracker."""
        if isinstance(d_value, Mobject):
            raise ValueError(
                "Cannot increment ValueTracker by a Mobject. Please provide a scalar value."
            )
        self.increment_value(d_value)
        return self

    def __floordiv__(self, d_value: float) -> ValueTracker:
        """Return a new :class:`ValueTracker` whose value is the floor division of the current
        tracker's value by ``d_value``.
        """
        return ValueTracker(self.get_value() // d_value)

    def __ifloordiv__(self, d_value: float) -> Self:
        """Set the value of this ValueTracker to the floor division of the current value by ``d_value``."""
        self.set_value(self.get_value() // d_value)
        return self

    def __mod__(self, d_value: float) -> ValueTracker:
        """Return a new :class:`ValueTracker` whose value is the current tracker's value
        modulo ``d_value``.
        """
        return ValueTracker(self.get_value() % d_value)

    def __imod__(self, d_value: float) -> Self:
        """Set the value of this ValueTracker to the current value modulo ``d_value``."""
        self.set_value(self.get_value() % d_value)
        return self

    def __mul__(self, d_value: float) -> ValueTracker:
        """Return a new :class:`ValueTracker` whose value is the current tracker's value multiplied by
        ``d_value``.
        """
        return ValueTracker(self.get_value() * d_value)

    def __imul__(self, d_value: float) -> Self:
        """Set the value of this ValueTracker to the product of the current value and ``d_value``."""
        self.set_value(self.get_value() * d_value)
        return self

    def __pow__(self, d_value: float) -> ValueTracker:
        """Return a new :class:`ValueTracker` whose value is the current tracker's value raised to the
        power of ``d_value``.
        """
        return ValueTracker(self.get_value() ** d_value)

    def __ipow__(self, d_value: float) -> Self:
        """Set the value of this ValueTracker to the current value raised to the power of ``d_value``."""
        self.set_value(self.get_value() ** d_value)
        return self

    def __sub__(self, d_value: float | Mobject) -> ValueTracker:
        """Return a new :class:`ValueTracker` whose value is the current tracker's value minus
        ``d_value``.
        """
        if isinstance(d_value, Mobject):
            raise ValueError(
                "Cannot decrement ValueTracker by a Mobject. Please provide a scalar value."
            )
        return ValueTracker(self.get_value() - d_value)

    def __isub__(self, d_value: float | Mobject) -> Self:
        """Adds ``-=`` syntax to decrement the value of the ValueTracker."""
        if isinstance(d_value, Mobject):
            raise ValueError(
                "Cannot decrement ValueTracker by a Mobject. Please provide a scalar value."
            )
        self.increment_value(-d_value)
        return self

    def __truediv__(self, d_value: float) -> ValueTracker:
        """Return a new :class:`ValueTracker` whose value is the current tracker's value
        divided by ``d_value``.
        """
        return ValueTracker(self.get_value() / d_value)

    def __itruediv__(self, d_value: float) -> Self:
        """Sets the value of this ValueTracker to the current value divided by ``d_value``."""
        self.set_value(self.get_value() / d_value)
        return self

    def interpolate(
        self,
        mobject1: Mobject,
        mobject2: Mobject,
        alpha: float,
        path_func: PathFuncType = straight_path(),
    ) -> Self:
        """Turns ``self`` into an interpolation between ``mobject1`` and ``mobject2``."""
        self.set(points=path_func(mobject1.points, mobject2.points, alpha))
        return self


class ComplexValueTracker(ValueTracker):
    """Tracks a complex-valued parameter.

    The value is internally stored as a points array [a, b, 0]. This can be accessed directly
    to represent the value geometrically, see the usage example.
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

    def get_value(self) -> complex:  # type: ignore [override]
        """Get the current value of this ComplexValueTracker as a complex number."""
        return complex(*self.points[0, :2])

    def set_value(
        self,
        value: complex | float | int | str | Sequence[float | int] | np.ndarray = 0+0j,
        mode: str = "rectangular",  # "rectangular" or "polar"
        angle_unit: str = "radians"  # "radians" or "degrees" — only used when mode="polar"
    ) -> Self:
        """
        Sets a new complex value to the ComplexValueTracker.

        Parameters
        ----------
        value : complex | float | int | str | Sequence[float | int] | np.ndarray
            The value to set. It can be:
            - a complex number: 2+3j
            - a float or int: 5.0 or 5
            - a valid numeric string: "23" or "2+3j"
            - a sequence of exactly 2 real numbers: (2, 3), [2, 3], np.array([2, 3])
                - if mode="rectangular": interpreted as (x, y)
                - if mode="polar": interpreted as (r, theta)
                    - theta can be in radians or degrees, specified by angle_unit
        mode : str
            "rectangular" (default) or "polar".
            Only relevant when value is a sequence.
        angle_unit : str
            "radians" (default) or "degrees".
            Only relevant when mode="polar".
            If "degrees", theta is converted to radians internally.

        Examples
        --------
        set_value(2+3j)                                    # rectangular complex
        set_value((2, 3))                                  # rectangular sequence
        set_value((1, 90), mode="polar", angle_unit="degrees")   # polar, degrees
        set_value((1, np.pi/2), mode="polar")              # polar, radians
        """

        # validate mode
        if mode not in ("rectangular", "polar"):
            raise ValueError(
                f"mode must be 'rectangular' or 'polar', got '{mode}'"
            )

        # validate angle_unit
        if angle_unit not in ("radians", "degrees"):
            raise ValueError(
                f"angle_unit must be 'radians' or 'degrees', got '{angle_unit}'"
            )

        if isinstance(value, (list, tuple, np.ndarray)):
            # length check
            if len(value) != 2:
                raise ValueError(
                    f"Expected exactly 2 numbers, got {len(value)}"
                )
            # check for type of number provided and finiteness check
            if not all(np.isreal(v) and np.isfinite(v) for v in value):
                raise TypeError(
                    f"Elements must be real and finite numbers — no NAN(Not a Number) or infinity is allowed"
                )
            a, b = value

            if mode == "polar":
                r, theta = a, b
                if r < 0:
                    raise ValueError(
                        f"Radius r must be non-negative in polar form, got {r}"
                    )
                # convert degrees to radians if needed
                if angle_unit == "degrees":
                    theta = np.deg2rad(theta)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
            else:  # rectangular
                x, y = a, b

        else:
            z = complex(value)  # handles complex, float, int, valid strings
            # check real and imag parts individually for finiteness
            if not np.isfinite(z.real):
                raise ValueError(
                    f"Real part must be finite, got {z.real}"
                )
            if not np.isfinite(z.imag):
                raise ValueError(
                    f"Imaginary part must be finite, got {z.imag}"
                )
            x, y = z.real, z.imag

        self.points[0, :2] = (x, y)
        return self