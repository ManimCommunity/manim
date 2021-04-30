"""Mobjects representing function graphs."""

__all__ = ["ParametricFunction", "FunctionGraph"]


import math

from .. import config
from ..constants import *
from ..mobject.types.vectorized_mobject import VMobject
from ..utils.color import YELLOW


class ParametricFunction(VMobject):
    """A parametric curve.

    Examples
    --------

    .. manim:: PlotParametricFunction
        :save_last_frame:

        class PlotParametricFunction(Scene):
            def func(self, t):
                return np.array((np.sin(2 * t), np.sin(3 * t), 0))

            def construct(self):
                func = ParametricFunction(self.func, t_max = TAU, fill_opacity=0).set_color(RED)
                self.add(func.scale(3))

    .. manim:: ThreeDParametricSpring
        :save_last_frame:

        class ThreeDParametricSpring(ThreeDScene):
            def construct(self):
                curve1 = ParametricFunction(
                    lambda u: np.array([
                        1.2 * np.cos(u),
                        1.2 * np.sin(u),
                        u * 0.05
                    ]), color=RED, t_min=-3 * TAU, t_max=5 * TAU,
                ).set_shade_in_3d(True)
                axes = ThreeDAxes()
                self.add(axes, curve1)
                self.set_camera_orientation(phi=80 * DEGREES, theta=-60 * DEGREES)
                self.wait()
    """

    def __init__(
        self,
        function=None,
        t_range=None,
        dt=1e-8,
        discontinuities=None,
        use_smoothing=True,
        **kwargs
    ):
        self.function = function
        t_range = np.array([0, 1, 0.01]) if t_range is None else t_range
        if len(t_range) == 2:
            t_range = [*t_range, 0.01]

        self.dt = dt
        self.discontinuities = [] if discontinuities is None else discontinuities
        self.use_smoothing = use_smoothing
        self.t_min, self.t_max, self.t_step = t_range

        VMobject.__init__(self, **kwargs)

    def get_function(self):
        return self.function

    def get_point_from_function(self, t):
        return self.function(t)

    def generate_points(self):

        discontinuities = filter(
            lambda t: self.t_min <= t <= self.t_max, self.discontinuities
        )
        discontinuities = np.array(list(discontinuities))
        boundary_times = [
            self.t_min,
            self.t_max,
            *(discontinuities - self.dt),
            *(discontinuities + self.dt),
        ]
        boundary_times.sort()
        for t1, t2 in zip(boundary_times[0::2], boundary_times[1::2]):
            t_range = [*np.arange(t1, t2, self.t_step), t2]
            points = np.array([self.function(t) for t in t_range])
            self.start_new_path(points[0])
            self.add_points_as_corners(points[1:])
        if self.use_smoothing:
            # TODO: not in line with upstream, approx_smooth does not exist
            self.make_smooth()
        return self


class FunctionGraph(ParametricFunction):
    def __init__(self, function, x_range=None, color=YELLOW, **kwargs):

        if x_range is None:
            x_range = np.array([-config["frame_x_radius"], config["frame_x_radius"]])

        self.x_range = x_range
        self.parametric_function = lambda t: np.array([t, function(t), 0])
        self.function = function
        super().__init__(self.parametric_function, self.x_range, color=color, **kwargs)

    def get_function(self):
        return self.function

    def get_point_from_function(self, x):
        return self.parametric_function(x)
