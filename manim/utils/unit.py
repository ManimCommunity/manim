"""Classes to convert between pixels, Manim units (Munits), degrees,
radians and percentage values as measurements in animations etc."""

from __future__ import annotations

import numpy as np

from .. import config, constants

__all__ = ["Pixels", "Degrees", "Munits", "Percent"]


class _PixelUnits:
    def __mul__(self, val):
        return val * config.frame_width / config.pixel_width

    def __rmul__(self, val):
        return val * config.frame_width / config.pixel_width


class Percent:
    """A percentage (0-100) scaling class of each axis.

    Parameters
    ----------
    axis
        The axis (X or Y) of the frame to consider the percentage of.
        Z_axis has no such frame limits so can't be used here.

    Examples
    --------

    .. manim :: PercentAxisExample
        :ref_classes: Axes Dot ApplyMethod

        class PercentAxisExample(Scene):
            def construct(self):
                axes = Axes(
                    x_range=[-1, 0, .1],
                    y_range=[-1, 0, .1],
                    x_length=100 * unit.Percent(X_AXIS),
                    y_length=100 * unit.Percent(Y_AXIS),
                    axis_config={"color": YELLOW},
                    tips=False
                )
                self.add(axes)

                dot = Dot(radius=0.2, color=RED)

                self.play(ApplyMethod(dot.shift, 30 * unit.Percent(X_AXIS) * RIGHT))
                self.play(ApplyMethod(dot.shift, 30 * unit.Percent(Y_AXIS) * UP))
                # using 30 and LEFT below achieves the same effect
                self.play(ApplyMethod(dot.shift, -30 * unit.Percent(X_AXIS) * RIGHT))
                # using 30 and DOWN below achieves the same effect
                self.play(ApplyMethod(dot.shift, -30 * unit.Percent(Y_AXIS) * UP))

    """

    def __init__(self, axis: np.ndarray):
        if np.array_equal(axis, constants.X_AXIS):
            self.length = config.frame_width
        if np.array_equal(axis, constants.Y_AXIS):
            self.length = config.frame_height
        if np.array_equal(axis, constants.Z_AXIS):
            raise NotImplementedError("length of Z axis is undefined")

    def __mul__(self, val: float) -> float:
        return val / 100 * self.length

    def __rmul__(self, val: float) -> float:
        return val / 100 * self.length


Pixels = _PixelUnits()
"""A scaling object to convert pixels to Munits

    Examples
    --------

    .. manim :: PixelsExample
        :ref_classes: Dot ApplyMethod

        class PixelsExample(Scene):
            def construct(self):
                dot1 = Dot(radius=0.2, color=RED).shift(UP)
                dot2 = Dot(radius=0.2, color=BLUE).shift(DOWN)
                self.play(
                    # With the standard Manim frame width/ratio,
                    # 150 Pixels = 2.5 Munits
                    ApplyMethod(dot2.shift, (150 * unit.Pixels) * RIGHT),
                    ApplyMethod(dot1.shift, 2.5 * RIGHT)
                )

"""

Degrees = constants.PI / 180
"""A convenience constant to convert degrees to radians
    Normal usage::

        Rotate(dot, 45 * unit.Degrees)  # equivalent to Rotate(dot, PI/4)

"""

Munits = 1
