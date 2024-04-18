"""Mobjects representing function graphs."""

from __future__ import annotations

__all__ = ["ParametricFunction", "FunctionGraph", "ImplicitFunction"]


from typing import TYPE_CHECKING, Callable, Iterable, Sequence

import numpy as np
from isosurfaces import plot_isoline

from manim import config
from manim.mobject.graphing.scale import LinearBase, _ScaleBase
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.types.vectorized_mobject import VMobject

if TYPE_CHECKING:
    from manim.typing import Point2D, Point3D

from manim.utils.color import YELLOW


class ParametricFunction(VMobject, metaclass=ConvertToOpenGL):
    """A parametric curve.

    Parameters
    ----------
    function
        The function to be plotted in the form of ``(lambda t: (x(t), y(t), z(t)))``
    t_range
        Determines the length that the function spans in the form of (t_min, t_max, step=0.01). By default ``[0, 1]``
    scaling
        Scaling class applied to the points of the function. Default of :class:`~.LinearBase`.
    use_smoothing
        Whether to interpolate between the points of the function after they have been created.
        (Will have odd behaviour with a low number of points)
    use_vectorized
        Whether to pass in the generated t value array to the function as ``[t_0, t_1, ...]``.
        Only use this if your function supports it. Output should be a numpy array
        of shape ``[[x_0, x_1, ...], [y_0, y_1, ...], [z_0, z_1, ...]]`` but ``z`` can
        also be 0 if the Axes is 2D
    discontinuities
        Values of t at which the function experiences discontinuity.
    dt
        The left and right tolerance for the discontinuities.


    Examples
    --------
    .. manim:: PlotParametricFunction
        :save_last_frame:

        class PlotParametricFunction(Scene):
            def func(self, t):
                return (np.sin(2 * t), np.sin(3 * t), 0)

            def construct(self):
                func = ParametricFunction(self.func, t_range = (0, TAU), fill_opacity=0).set_color(RED)
                self.add(func.scale(3))

    .. manim:: ThreeDParametricSpring
        :save_last_frame:

        class ThreeDParametricSpring(ThreeDScene):
            def construct(self):
                curve1 = ParametricFunction(
                    lambda u: (
                        1.2 * np.cos(u),
                        1.2 * np.sin(u),
                        u * 0.05
                    ), color=RED, t_range = (-3*TAU, 5*TAU, 0.01)
                ).set_shade_in_3d(True)
                axes = ThreeDAxes()
                self.add(axes, curve1)
                self.set_camera_orientation(phi=80 * DEGREES, theta=-60 * DEGREES)
                self.wait()

    .. attention::
        If your function has discontinuities, you'll have to specify the location
        of the discontinuities manually. See the following example for guidance.

    .. manim:: DiscontinuousExample
        :save_last_frame:

        class DiscontinuousExample(Scene):
            def construct(self):
                ax1 = NumberPlane((-3, 3), (-4, 4))
                ax2 = NumberPlane((-3, 3), (-4, 4))
                VGroup(ax1, ax2).arrange()
                discontinuous_function = lambda x: (x ** 2 - 2) / (x ** 2 - 4)
                incorrect = ax1.plot(discontinuous_function, color=RED)
                correct = ax2.plot(
                    discontinuous_function,
                    discontinuities=[-2, 2],  # discontinuous points
                    dt=0.1,  # left and right tolerance of discontinuity
                    color=GREEN,
                )
                self.add(ax1, ax2, incorrect, correct)
    """

    def __init__(
        self,
        function: Callable[[float], Point3D],
        t_range: Point2D | Point3D = (0, 1),
        scaling: _ScaleBase = LinearBase(),
        dt: float = 1e-8,
        discontinuities: Iterable[float] | None = None,
        use_smoothing: bool = True,
        use_vectorized: bool = False,
        **kwargs,
    ):
        self.function = function
        t_range = (0, 1, 0.01) if t_range is None else t_range
        if len(t_range) == 2:
            t_range = np.array([*t_range, 0.01])

        self.scaling = scaling

        self.dt = dt
        self.discontinuities = discontinuities
        self.use_smoothing = use_smoothing
        self.use_vectorized = use_vectorized
        self.t_min, self.t_max, self.t_step = t_range

        super().__init__(**kwargs)

    def get_function(self):
        return self.function

    def get_point_from_function(self, t):
        return self.function(t)

    def generate_points(self):
        if self.discontinuities is not None:
            discontinuities = filter(
                lambda t: self.t_min <= t <= self.t_max,
                self.discontinuities,
            )
            discontinuities = np.array(list(discontinuities))
            boundary_times = np.array(
                [
                    self.t_min,
                    self.t_max,
                    *(discontinuities - self.dt),
                    *(discontinuities + self.dt),
                ],
            )
            boundary_times.sort()
        else:
            boundary_times = [self.t_min, self.t_max]

        for t1, t2 in zip(boundary_times[0::2], boundary_times[1::2]):
            t_range = np.array(
                [
                    *self.scaling.function(np.arange(t1, t2, self.t_step)),
                    self.scaling.function(t2),
                ],
            )

            if self.use_vectorized:
                x, y, z = self.function(t_range)
                if not isinstance(z, np.ndarray):
                    z = np.zeros_like(x)
                points = np.stack([x, y, z], axis=1)
            else:
                points = np.array([self.function(t) for t in t_range])

            self.start_new_path(points[0])
            self.add_points_as_corners(points[1:])
        if self.use_smoothing:
            # TODO: not in line with upstream, approx_smooth does not exist
            self.make_smooth()
        return self

    init_points = generate_points


class FunctionGraph(ParametricFunction):
    """A :class:`ParametricFunction` that spans the length of the scene by default.

    Examples
    --------
    .. manim:: ExampleFunctionGraph
        :save_last_frame:

        class ExampleFunctionGraph(Scene):
            def construct(self):
                cos_func = FunctionGraph(
                    lambda t: np.cos(t) + 0.5 * np.cos(7 * t) + (1 / 7) * np.cos(14 * t),
                    color=RED,
                )

                sin_func_1 = FunctionGraph(
                    lambda t: np.sin(t) + 0.5 * np.sin(7 * t) + (1 / 7) * np.sin(14 * t),
                    color=BLUE,
                )

                sin_func_2 = FunctionGraph(
                    lambda t: np.sin(t) + 0.5 * np.sin(7 * t) + (1 / 7) * np.sin(14 * t),
                    x_range=[-4, 4],
                    color=GREEN,
                ).move_to([0, 1, 0])

                self.add(cos_func, sin_func_1, sin_func_2)
    """

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


class ImplicitFunction(VMobject, metaclass=ConvertToOpenGL):
    def __init__(
        self,
        func: Callable[[float, float], float],
        x_range: Sequence[float] | None = None,
        y_range: Sequence[float] | None = None,
        min_depth: int = 5,
        max_quads: int = 1500,
        use_smoothing: bool = True,
        **kwargs,
    ):
        """An implicit function.

        Parameters
        ----------
        func
            The implicit function in the form ``f(x, y) = 0``.
        x_range
            The x min and max of the function.
        y_range
            The y min and max of the function.
        min_depth
            The minimum depth of the function to calculate.
        max_quads
            The maximum number of quads to use.
        use_smoothing
            Whether or not to smoothen the curves.
        kwargs
            Additional parameters to pass into :class:`VMobject`


        .. note::
            A small ``min_depth`` :math:`d` means that some small details might
            be ignored if they don't cross an edge of one of the
            :math:`4^d` uniform quads.

            The value of ``max_quads`` strongly corresponds to the
            quality of the curve, but a higher number of quads
            may take longer to render.

        Examples
        --------
        .. manim:: ImplicitFunctionExample
            :save_last_frame:

            class ImplicitFunctionExample(Scene):
                def construct(self):
                    graph = ImplicitFunction(
                        lambda x, y: x * y ** 2 - x ** 2 * y - 2,
                        color=YELLOW
                    )
                    self.add(NumberPlane(), graph)
        """
        self.function = func
        self.min_depth = min_depth
        self.max_quads = max_quads
        self.use_smoothing = use_smoothing
        self.x_range = x_range or [
            -config.frame_width / 2,
            config.frame_width / 2,
        ]
        self.y_range = y_range or [
            -config.frame_height / 2,
            config.frame_height / 2,
        ]

        super().__init__(**kwargs)

    def generate_points(self):
        p_min, p_max = (
            np.array([self.x_range[0], self.y_range[0]]),
            np.array([self.x_range[1], self.y_range[1]]),
        )
        curves = plot_isoline(
            fn=lambda u: self.function(u[0], u[1]),
            pmin=p_min,
            pmax=p_max,
            min_depth=self.min_depth,
            max_quads=self.max_quads,
        )  # returns a list of lists of 2D points
        curves = [
            np.pad(curve, [(0, 0), (0, 1)]) for curve in curves if curve != []
        ]  # add z coord as 0
        for curve in curves:
            self.start_new_path(curve[0])
            self.add_points_as_corners(curve[1:])
        if self.use_smoothing:
            self.make_smooth()
        return self

    init_points = generate_points
