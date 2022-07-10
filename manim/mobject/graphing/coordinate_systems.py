"""Mobjects that represent coordinate systems."""


from __future__ import annotations

__all__ = [
    "CoordinateSystem",
    "Axes",
    "ThreeDAxes",
    "NumberPlane",
    "PolarPlane",
    "ComplexPlane",
]

import fractions as fr
import numbers
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import numpy as np
from colour import Color

from manim import config
from manim.constants import *
from manim.mobject.geometry.arc import Circle, Dot
from manim.mobject.geometry.line import Arrow, DashedLine, Line
from manim.mobject.geometry.polygram import Polygon, Rectangle, RegularPolygon
from manim.mobject.graphing.functions import ImplicitFunction, ParametricFunction
from manim.mobject.graphing.number_line import NumberLine
from manim.mobject.graphing.scale import LinearBase
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.text.tex_mobject import MathTex
from manim.mobject.types.vectorized_mobject import (
    VDict,
    VectorizedPoint,
    VGroup,
    VMobject,
)
from manim.utils.color import (
    BLACK,
    BLUE,
    BLUE_D,
    GREEN,
    WHITE,
    YELLOW,
    color_gradient,
    invert_color,
)
from manim.utils.config_ops import merge_dicts_recursively, update_dict_recursively
from manim.utils.simple_functions import binary_search
from manim.utils.space_ops import angle_of_vector

if TYPE_CHECKING:
    from manim.mobject.mobject import Mobject


class CoordinateSystem:
    r"""Abstract base class for Axes and NumberPlane.

    Examples
    --------
    .. manim:: CoordSysExample
        :save_last_frame:

        class CoordSysExample(Scene):
            def construct(self):
                # the location of the ticks depends on the x_range and y_range.
                grid = Axes(
                    x_range=[0, 1, 0.05],  # step size determines num_decimal_places.
                    y_range=[0, 1, 0.05],
                    x_length=9,
                    y_length=5.5,
                    axis_config={
                        "numbers_to_include": np.arange(0, 1 + 0.1, 0.1),
                        "font_size": 24,
                    },
                    tips=False,
                )

                # Labels for the x-axis and y-axis.
                y_label = grid.get_y_axis_label("y", edge=LEFT, direction=LEFT, buff=0.4)
                x_label = grid.get_x_axis_label("x")
                grid_labels = VGroup(x_label, y_label)

                graphs = VGroup()
                for n in np.arange(1, 20 + 0.5, 0.5):
                    graphs += grid.plot(lambda x: x ** n, color=WHITE)
                    graphs += grid.plot(
                        lambda x: x ** (1 / n), color=WHITE, use_smoothing=False
                    )

                # Extra lines and labels for point (1,1)
                graphs += grid.get_horizontal_line(grid.c2p(1, 1, 0), color=BLUE)
                graphs += grid.get_vertical_line(grid.c2p(1, 1, 0), color=BLUE)
                graphs += Dot(point=grid.c2p(1, 1, 0), color=YELLOW)
                graphs += Tex("(1,1)").scale(0.75).next_to(grid.c2p(1, 1, 0))
                title = Title(
                    # spaces between braces to prevent SyntaxError
                    r"Graphs of $y=x^{ {1}\over{n} }$ and $y=x^n (n=1,2,3,...,20)$",
                    include_underline=False,
                    font_size=40,
                )

                self.add(title, graphs, grid, grid_labels)
    """

    def __init__(
        self,
        x_range=None,
        y_range=None,
        x_length=None,
        y_length=None,
        dimension=2,
    ):
        self.dimension = dimension

        default_step = 1
        if x_range is None:
            x_range = [
                round(-config["frame_x_radius"]),
                round(config["frame_x_radius"]),
                default_step,
            ]
        elif len(x_range) == 2:
            x_range = [*x_range, default_step]

        if y_range is None:
            y_range = [
                round(-config["frame_y_radius"]),
                round(config["frame_y_radius"]),
                default_step,
            ]
        elif len(y_range) == 2:
            y_range = [*y_range, default_step]

        self.x_range = x_range
        self.y_range = y_range
        self.x_length = x_length
        self.y_length = y_length
        self.num_sampled_graph_points_per_tick = 10

    def coords_to_point(self, *coords):
        raise NotImplementedError()

    def point_to_coords(self, point):
        raise NotImplementedError()

    def polar_to_point(self, radius: float, azimuth: float) -> np.ndarray:
        r"""Gets a point from polar coordinates.

        Parameters
        ----------
        radius
            The coordinate radius (:math:`r`).

        azimuth
            The coordinate azimuth (:math:`\theta`).

        Returns
        -------
        numpy.ndarray
            The point.

        Examples
        --------
        .. manim:: PolarToPointExample
            :ref_classes: PolarPlane Vector
            :save_last_frame:

            class PolarToPointExample(Scene):
                def construct(self):
                    polarplane_pi = PolarPlane(azimuth_units="PI radians", size=6)
                    polartopoint_vector = Vector(polarplane_pi.polar_to_point(3, PI/4))
                    self.add(polarplane_pi)
                    self.add(polartopoint_vector)
        """
        return self.coords_to_point(radius * np.cos(azimuth), radius * np.sin(azimuth))

    def point_to_polar(self, point: np.ndarray) -> tuple[float, float]:
        r"""Gets polar coordinates from a point.

        Parameters
        ----------
        point
            The point.

        Returns
        -------
        Tuple[:class:`float`, :class:`float`]
            The coordinate radius (:math:`r`) and the coordinate azimuth (:math:`\theta`).
        """
        x, y = self.point_to_coords(point)
        return np.sqrt(x**2 + y**2), np.arctan2(y, x)

    def c2p(self, *coords):
        """Abbreviation for :meth:`coords_to_point`"""
        return self.coords_to_point(*coords)

    def p2c(self, point):
        """Abbreviation for :meth:`point_to_coords`"""
        return self.point_to_coords(point)

    def pr2pt(self, radius: float, azimuth: float) -> np.ndarray:
        """Abbreviation for :meth:`polar_to_point`"""
        return self.polar_to_point(radius, azimuth)

    def pt2pr(self, point: np.ndarray) -> tuple[float, float]:
        """Abbreviation for :meth:`point_to_polar`"""
        return self.point_to_polar(point)

    def get_axes(self):
        raise NotImplementedError()

    def get_axis(self, index):
        return self.get_axes()[index]

    def get_origin(self) -> np.ndarray:
        """Gets the origin of :class:`~.Axes`.

        Returns
        -------
        np.ndarray
            The center point.
        """
        return self.coords_to_point(0, 0)

    def get_x_axis(self):
        return self.get_axis(0)

    def get_y_axis(self):
        return self.get_axis(1)

    def get_z_axis(self):
        return self.get_axis(2)

    def get_x_unit_size(self):
        return self.get_x_axis().get_unit_size()

    def get_y_unit_size(self):
        return self.get_y_axis().get_unit_size()

    def get_x_axis_label(
        self,
        label: float | str | Mobject,
        edge: Sequence[float] = UR,
        direction: Sequence[float] = UR,
        buff: float = SMALL_BUFF,
        **kwargs,
    ) -> Mobject:
        """Generate an x-axis label.

        Parameters
        ----------
        label
            The label. Defaults to :class:`~.MathTex` for ``str`` and ``float`` inputs.
        edge
            The edge of the x-axis to which the label will be added, by default ``UR``.
        direction
            Allows for further positioning of the label from an edge, by default ``UR``.
        buff
            The distance of the label from the line.

        Returns
        -------
        :class:`~.Mobject`
            The positioned label.

        Examples
        --------
        .. manim:: GetXAxisLabelExample
            :save_last_frame:

            class GetXAxisLabelExample(Scene):
                def construct(self):
                    ax = Axes(x_range=(0, 8), y_range=(0, 5), x_length=8, y_length=5)
                    x_label = ax.get_x_axis_label(
                        Tex("$x$-values").scale(0.65), edge=DOWN, direction=DOWN, buff=0.5
                    )
                    self.add(ax, x_label)
        """
        return self._get_axis_label(
            label, self.get_x_axis(), edge, direction, buff=buff, **kwargs
        )

    def get_y_axis_label(
        self,
        label: float | str | Mobject,
        edge: Sequence[float] = UR,
        direction: Sequence[float] = UP * 0.5 + RIGHT,
        buff: float = SMALL_BUFF,
        **kwargs,
    ):
        """Generate a y-axis label.

        Parameters
        ----------
        label
            The label. Defaults to :class:`~.MathTex` for ``str`` and ``float`` inputs.
        edge
            The edge of the x-axis to which the label will be added, by default ``UR``.
        direction
            Allows for further positioning of the label from an edge, by default ``UR``
        buff
            The distance of the label from the line.

        Returns
        -------
        :class:`~.Mobject`
            The positioned label.

        Examples
        --------
        .. manim:: GetYAxisLabelExample
            :save_last_frame:

            class GetYAxisLabelExample(Scene):
                def construct(self):
                    ax = Axes(x_range=(0, 8), y_range=(0, 5), x_length=8, y_length=5)
                    y_label = ax.get_y_axis_label(
                        Tex("$y$-values").scale(0.65).rotate(90 * DEGREES),
                        edge=LEFT,
                        direction=LEFT,
                        buff=0.3,
                    )
                    self.add(ax, y_label)
        """

        return self._get_axis_label(
            label, self.get_y_axis(), edge, direction, buff=buff, **kwargs
        )

    def _get_axis_label(
        self,
        label: float | str | Mobject,
        axis: Mobject,
        edge: Sequence[float],
        direction: Sequence[float],
        buff: float = SMALL_BUFF,
    ) -> Mobject:
        """Gets the label for an axis.

        Parameters
        ----------
        label
            The label. Defaults to :class:`~.MathTex` for ``str`` and ``float`` inputs.
        axis
            The axis to which the label will be added.
        edge
            The edge of the axes to which the label will be added. ``RIGHT`` adds to the right side of the axis
        direction
            Allows for further positioning of the label.
        buff
            The distance of the label from the line.

        Returns
        -------
        :class:`~.Mobject`
            The positioned label along the given axis.
        """

        label = self.x_axis._create_label_tex(label)
        label.next_to(axis.get_edge_center(edge), direction=direction, buff=buff)
        label.shift_onto_screen(buff=MED_SMALL_BUFF)
        return label

    def get_axis_labels(
        self,
        x_label: float | str | Mobject = "x",
        y_label: float | str | Mobject = "y",
    ) -> VGroup:
        """Defines labels for the x_axis and y_axis of the graph.

        For increased control over the position of the labels,
        use :meth:`get_x_axis_label` and :meth:`get_y_axis_label`.

        Parameters
        ----------
        x_label
            The label for the x_axis. Defaults to :class:`~.MathTex` for ``str`` and ``float`` inputs.
        y_label
            The label for the y_axis. Defaults to :class:`~.MathTex` for ``str`` and ``float`` inputs.

        Returns
        -------
        :class:`~.VGroup`
            A :class:`~.Vgroup` of the labels for the x_axis and y_axis.


        .. seealso::
            :class:`get_x_axis_label`
            :class:`get_y_axis_label`

        Examples
        --------
        .. manim:: GetAxisLabelsExample
            :save_last_frame:

            class GetAxisLabelsExample(Scene):
                def construct(self):
                    ax = Axes()
                    labels = ax.get_axis_labels(
                        Tex("x-axis").scale(0.7), Text("y-axis").scale(0.45)
                    )
                    self.add(ax, labels)
        """

        self.axis_labels = VGroup(
            self.get_x_axis_label(x_label),
            self.get_y_axis_label(y_label),
        )
        return self.axis_labels

    def add_coordinates(
        self,
        *axes_numbers: (Iterable[float] | None | dict[float, str | float | Mobject]),
        **kwargs,
    ):
        """Adds labels to the axes. Use ``Axes.coordinate_labels`` to
        access the coordinates after creation.

        Parameters
        ----------
        axes_numbers
            The numbers to be added to the axes. Use ``None`` to represent an axis with default labels.

        Examples
        --------
        .. code-block:: python

            ax = ThreeDAxes()
            x_labels = range(-4, 5)
            z_labels = range(-4, 4, 2)
            ax.add_coordinates(x_labels, None, z_labels)  # default y labels, custom x & z labels
            ax.add_coordinates(x_labels)  # only x labels

        You can also specifically control the position and value of the labels using a dict.

        .. code-block:: python

            ax = Axes(x_range=[0, 7])
            x_pos = [x for x in range(1, 8)]

            # strings are automatically converted into a Tex mobject.
            x_vals = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            x_dict = dict(zip(x_pos, x_vals))
            ax.add_coordinates(x_dict)
        """

        self.coordinate_labels = VGroup()
        # if nothing is passed to axes_numbers, produce axes with default labelling
        if not axes_numbers:
            axes_numbers = [None for _ in range(self.dimension)]

        for axis, values in zip(self.axes, axes_numbers):
            if isinstance(values, dict):
                axis.add_labels(values, **kwargs)
                labels = axis.labels
            elif values is None and axis.scaling.custom_labels:
                tick_range = axis.get_tick_range()
                axis.add_labels(
                    dict(zip(tick_range, axis.scaling.get_custom_labels(tick_range)))
                )
                labels = axis.labels
            else:
                axis.add_numbers(values, **kwargs)
                labels = axis.numbers
            self.coordinate_labels.add(labels)

        return self

    def get_line_from_axis_to_point(
        self,
        index: int,
        point: Sequence[float],
        line_func: Line = DashedLine,
        line_config: dict | None = None,
        color: Color | None = None,
        stroke_width: float = 2,
    ) -> Line:
        """Returns a straight line from a given axis to a point in the scene.

        Parameters
        ----------
        index
            Specifies the axis from which to draw the line. `0 = x_axis`, `1 = y_axis`
        point
            The point to which the line will be drawn.
        line_func
            The function of the :class:`~.Line` mobject used to construct the line.
        line_config
            Optional arguments to passed to :attr:`line_func`.
        color
            The color of the line.
        stroke_width
            The stroke width of the line.

        Returns
        -------
        :class:`~.Line`
            The line from an axis to a point.


        .. seealso::
            :meth:`~.CoordinateSystem.get_vertical_line`
            :meth:`~.CoordinateSystem.get_horizontal_line`
        """

        line_config = line_config if line_config is not None else {}

        if color is None:
            color = VMobject().color

        line_config["color"] = color
        line_config["stroke_width"] = stroke_width

        axis = self.get_axis(index)
        line = line_func(axis.get_projection(point), point, **line_config)
        return line

    def get_vertical_line(self, point: Sequence[float], **kwargs) -> Line:
        """A vertical line from the x-axis to a given point in the scene.

        Parameters
        ----------
        point
            The point to which the vertical line will be drawn.
        kwargs
            Additional parameters to be passed to :class:`get_line_from_axis_to_point`.

        Returns
        -------
        :class:`Line`
            A vertical line from the x-axis to the point.

        Examples
        --------
        .. manim:: GetVerticalLineExample
            :save_last_frame:

            class GetVerticalLineExample(Scene):
                def construct(self):
                    ax = Axes().add_coordinates()
                    point = ax.coords_to_point(-3.5, 2)

                    dot = Dot(point)
                    line = ax.get_vertical_line(point, line_config={"dashed_ratio": 0.85})

                    self.add(ax, line, dot)


        """
        return self.get_line_from_axis_to_point(0, point, **kwargs)

    def get_horizontal_line(self, point: Sequence[float], **kwargs) -> Line:
        """A horizontal line from the y-axis to a given point in the scene.

        Parameters
        ----------
        point
            The point to which the horizontal line will be drawn.
        kwargs
            Additional parameters to be passed to :class:`get_line_from_axis_to_point`.

        Returns
        -------
        :class:`Line`
            A horizontal line from the y-axis to the point.

        Examples
        --------
        .. manim:: GetHorizontalLineExample
            :save_last_frame:

            class GetHorizontalLineExample(Scene):
                def construct(self):
                    ax = Axes().add_coordinates()
                    point = ax.c2p(-4, 1.5)

                    dot = Dot(point)
                    line = ax.get_horizontal_line(point, line_func=Line)

                    self.add(ax, line, dot)
        """

        return self.get_line_from_axis_to_point(1, point, **kwargs)

    def get_lines_to_point(self, point: Sequence[float], **kwargs) -> VGroup:
        """Generate both horizontal and vertical lines from the axis to a point.

        Parameters
        ----------
        point
            A point on the scene.
        kwargs
            Additional parameters to be passed to :meth:`get_line_from_axis_to_point`

        Returns
        -------
        :class:`~.VGroup`
            A :class:`~.VGroup` of the horizontal and vertical lines.


        .. seealso::
            :meth:`~.CoordinateSystem.get_vertical_line`
            :meth:`~.CoordinateSystem.get_horizontal_line`

        Examples
        --------
        .. manim:: GetLinesToPointExample
            :save_last_frame:

            class GetLinesToPointExample(Scene):
                def construct(self):
                    ax = Axes()
                    circ = Circle(radius=0.5).move_to([-4, -1.5, 0])

                    lines_1 = ax.get_lines_to_point(circ.get_right(), color=GREEN_B)
                    lines_2 = ax.get_lines_to_point(circ.get_corner(DL), color=BLUE_B)
                    self.add(ax, lines_1, lines_2, circ)
        """

        return VGroup(
            self.get_horizontal_line(point, **kwargs),
            self.get_vertical_line(point, **kwargs),
        )

    # graphing

    def plot(
        self,
        function: Callable[[float], float],
        x_range: Sequence[float] | None = None,
        use_vectorized: bool = False,
        **kwargs,
    ):
        """Generates a curve based on a function.

        Parameters
        ----------
        function
            The function used to construct the :class:`~.ParametricFunction`.
        x_range
            The range of the curve along the axes. ``x_range = [x_min, x_max, x_step]``.
        use_vectorized
            Whether to pass in the generated t value array to the function. Only use this if your function supports it.
            Output should be a numpy array of shape ``[y_0, y_1, ...]``
        kwargs
            Additional parameters to be passed to :class:`~.ParametricFunction`.

        Returns
        -------
        :class:`~.ParametricFunction`
            The plotted curve.


        .. warning::
            This method may not produce accurate graphs since Manim currently relies on interpolation between
            evenly-spaced samples of the curve, instead of intelligent plotting.
            See the example below for some solutions to this problem.

        Examples
        --------
        .. manim:: PlotExample
            :save_last_frame:

            class PlotExample(Scene):
                def construct(self):
                    # construct the axes
                    ax_1 = Axes(
                        x_range=[0.001, 6],
                        y_range=[-8, 2],
                        x_length=5,
                        y_length=3,
                        tips=False,
                    )
                    ax_2 = ax_1.copy()
                    ax_3 = ax_1.copy()

                    # position the axes
                    ax_1.to_corner(UL)
                    ax_2.to_corner(UR)
                    ax_3.to_edge(DOWN)
                    axes = VGroup(ax_1, ax_2, ax_3)

                    # create the logarithmic curves
                    def log_func(x):
                        return np.log(x)

                    # a curve without adjustments; poor interpolation.
                    curve_1 = ax_1.plot(log_func, color=PURE_RED)

                    # disabling interpolation makes the graph look choppy as not enough
                    # inputs are available
                    curve_2 = ax_2.plot(log_func, use_smoothing=False, color=ORANGE)

                    # taking more inputs of the curve by specifying a step for the
                    # x_range yields expected results, but increases rendering time.
                    curve_3 = ax_3.plot(
                        log_func, x_range=(0.001, 6, 0.001), color=PURE_GREEN
                    )

                    curves = VGroup(curve_1, curve_2, curve_3)

                    self.add(axes, curves)
        """

        t_range = np.array(self.x_range, dtype=float)
        if x_range is not None:
            t_range[: len(x_range)] = x_range

        if x_range is None or len(x_range) < 3:
            # if t_range has a defined step size, increase the number of sample points per tick
            t_range[2] /= self.num_sampled_graph_points_per_tick
        # For axes, the third coordinate of x_range indicates
        # tick frequency.  But for functions, it indicates a
        # sample frequency

        graph = ParametricFunction(
            lambda t: self.coords_to_point(t, function(t)),
            t_range=t_range,
            scaling=self.x_axis.scaling,
            use_vectorized=use_vectorized,
            **kwargs,
        )
        graph.underlying_function = function
        return graph

    def plot_implicit_curve(
        self,
        func: Callable,
        min_depth: int = 5,
        max_quads: int = 1500,
        **kwargs,
    ) -> ImplicitFunction:
        """Creates the curves of an implicit function.

        Parameters
        ----------
        func
            The function to graph, in the form of f(x, y) = 0.
        min_depth
            The minimum depth of the function to calculate.
        max_quads
            The maximum number of quads to use.
        kwargs
            Additional parameters to pass into :class:`ImplicitFunction`.

        Examples
        --------
        .. manim:: ImplicitExample
            :save_last_frame:

            class ImplicitExample(Scene):
                def construct(self):
                    ax = Axes()
                    a = ax.plot_implicit_curve(
                        lambda x, y: y * (x - y) ** 2 - 4 * x - 8, color=BLUE
                    )
                    self.add(ax, a)
        """
        graph = ImplicitFunction(
            func=func,
            x_range=self.x_range[:2],
            y_range=self.y_range[:2],
            min_depth=min_depth,
            max_quads=max_quads,
            **kwargs,
        )
        (
            graph.stretch(self.get_x_unit_size(), 0, about_point=ORIGIN)
            .stretch(self.get_y_unit_size(), 1, about_point=ORIGIN)
            .shift(self.get_origin())
        )
        return graph

    def plot_parametric_curve(
        self,
        function: Callable[[float], np.ndarray],
        use_vectorized: bool = False,
        **kwargs,
    ) -> ParametricFunction:
        """A parametric curve.

        Parameters
        ----------
        function
            A parametric function mapping a number to a point in the
            coordinate system.
        use_vectorized
            Whether to pass in the generated t value array to the function. Only use this if your function supports it.
        kwargs
            Any further keyword arguments are passed to :class:`.ParametricFunction`.

        Example
        -------
        .. manim:: ParametricCurveExample
            :save_last_frame:

            class ParametricCurveExample(Scene):
                def construct(self):
                    ax = Axes()
                    cardioid = ax.plot_parametric_curve(
                        lambda t: np.array(
                            [
                                np.exp(1) * np.cos(t) * (1 - np.cos(t)),
                                np.exp(1) * np.sin(t) * (1 - np.cos(t)),
                                0,
                            ]
                        ),
                        t_range=[0, 2 * PI],
                        color="#0FF1CE",
                    )
                    self.add(ax, cardioid)
        """
        dim = self.dimension
        graph = ParametricFunction(
            lambda t: self.coords_to_point(*function(t)[:dim]),
            use_vectorized=use_vectorized,
            **kwargs,
        )
        graph.underlying_function = function
        return graph

    def plot_polar_graph(
        self,
        r_func: Callable[[float], float],
        theta_range: Sequence[float] = [0, 2 * PI],
        **kwargs,
    ) -> ParametricFunction:
        """A polar graph.

        Parameters
        ----------
        r_func
            The function r of theta.
        theta_range
            The range of theta as ``theta_range = [theta_min, theta_max, theta_step]``.
        kwargs
            Additional parameters passed to :class:`~.ParametricFunction`.

        Examples
        --------
        .. manim:: PolarGraphExample
            :ref_classes: PolarPlane
            :save_last_frame:

            class PolarGraphExample(Scene):
                def construct(self):
                    plane = PolarPlane()
                    r = lambda theta: 2 * np.sin(theta * 5)
                    graph = plane.plot_polar_graph(r, [0, 2 * PI], color=ORANGE)
                    self.add(plane, graph)
        """
        graph = ParametricFunction(
            function=lambda th: self.pr2pt(r_func(th), th),
            t_range=theta_range,
            **kwargs,
        )
        graph.underlying_function = r_func
        return graph

    def input_to_graph_point(
        self,
        x: float,
        graph: ParametricFunction | VMobject,
    ) -> np.ndarray:
        """Returns the coordinates of the point on a ``graph`` corresponding to an ``x`` value.

        Parameters
        ----------
        x
            The x-value of a point on the ``graph``.
        graph
            The :class:`~.ParametricFunction` on which the point lies.

        Returns
        -------
        :class:`np.ndarray`
            The coordinates of the point on the :attr:`graph` corresponding to the :attr:`x` value.

        Raises
        ------
        :exc:`ValueError`
            When the target x is not in the range of the line graph.

        Examples
        --------
        .. manim:: InputToGraphPointExample
            :save_last_frame:

            class InputToGraphPointExample(Scene):
                def construct(self):
                    ax = Axes()
                    curve = ax.plot(lambda x : np.cos(x))

                    # move a square to PI on the cosine curve.
                    position = ax.input_to_graph_point(x=PI, graph=curve)
                    sq = Square(side_length=1, color=YELLOW).move_to(position)

                    self.add(ax, curve, sq)
        """

        if hasattr(graph, "underlying_function"):
            return graph.function(x)
        else:
            alpha = binary_search(
                function=lambda a: self.point_to_coords(graph.point_from_proportion(a))[
                    0
                ],
                target=x,
                lower_bound=0,
                upper_bound=1,
            )
            if alpha is not None:
                return graph.point_from_proportion(alpha)
            else:
                raise ValueError(
                    f"x={x} not located in the range of the graph ([{self.p2c(graph.get_start())[0]}, {self.p2c(graph.get_end())[0]}])",
                )

    def input_to_graph_coords(self, x: float, graph: ParametricFunction) -> tuple:
        """Returns a tuple of the axis relative coordinates of the point
        on the graph based on the x-value given.

        Examples
        --------
        .. code-block:: pycon

            >>> from manim import Axes
            >>> ax = Axes()
            >>> parabola = ax.plot(lambda x: x**2)
            >>> ax.input_to_graph_coords(x=3, graph=parabola)
            (3, 9)
        """
        return x, graph.underlying_function(x)

    def i2gc(self, x: float, graph: ParametricFunction) -> tuple:
        """Alias for :meth:`input_to_graph_coords`."""
        return self.input_to_graph_coords(x, graph)

    def i2gp(self, x: float, graph: ParametricFunction) -> np.ndarray:
        """Alias for :meth:`input_to_graph_point`."""
        return self.input_to_graph_point(x, graph)

    def get_graph_label(
        self,
        graph: ParametricFunction,
        label: float | str | Mobject = "f(x)",
        x_val: float | None = None,
        direction: Sequence[float] = RIGHT,
        buff: float = MED_SMALL_BUFF,
        color: Color | None = None,
        dot: bool = False,
        dot_config: dict | None = None,
    ) -> Mobject:
        """Creates a properly positioned label for the passed graph, with an optional dot.

        Parameters
        ----------
        graph
            The curve.
        label
            The label for the function's curve. Defaults to :class:`~.MathTex` for ``str`` and ``float`` inputs.
        x_val
            The x_value along the curve that positions the label.
        direction
            The cartesian position, relative to the curve that the label will be at --> ``LEFT``, ``RIGHT``.
        buff
            The distance between the curve and the label.
        color
            The color of the label. Defaults to the color of the curve.
        dot
            Whether to add a dot at the point on the graph.
        dot_config
            Additional parameters to be passed to :class:`~.Dot`.

        Returns
        -------
        :class:`Mobject`
            The positioned label and :class:`~.Dot`, if applicable.

        Examples
        --------
        .. manim:: GetGraphLabelExample
            :save_last_frame:

            class GetGraphLabelExample(Scene):
                def construct(self):
                    ax = Axes()
                    sin = ax.plot(lambda x: np.sin(x), color=PURPLE_B)
                    label = ax.get_graph_label(
                        graph=sin,
                        label= MathTex(r"\\frac{\\pi}{2}"),
                        x_val=PI / 2,
                        dot=True,
                        direction=UR,
                    )

                    self.add(ax, sin, label)
        """

        if dot_config is None:
            dot_config = {}
        color = color or graph.get_color()
        label = self.x_axis._create_label_tex(label).set_color(color)

        if x_val is None:
            # Search from right to left
            for x in np.linspace(self.x_range[1], self.x_range[0], 100):
                point = self.input_to_graph_point(x, graph)
                if point[1] < config["frame_y_radius"]:
                    break
        else:
            point = self.input_to_graph_point(x_val, graph)

        label.next_to(point, direction, buff=buff)
        label.shift_onto_screen()

        if dot:
            dot = Dot(point=point, **dot_config)
            label.add(dot)
            label.dot = dot
        return label

    # calculus

    def get_riemann_rectangles(
        self,
        graph: ParametricFunction,
        x_range: Sequence[float] | None = None,
        dx: float | None = 0.1,
        input_sample_type: str = "left",
        stroke_width: float = 1,
        stroke_color: Color = BLACK,
        fill_opacity: float = 1,
        color: Iterable[Color] | Color = np.array((BLUE, GREEN)),
        show_signed_area: bool = True,
        bounded_graph: ParametricFunction = None,
        blend: bool = False,
        width_scale_factor: float = 1.001,
    ) -> VGroup:
        """Generates a :class:`~.VGroup` of the Riemann Rectangles for a given curve.

        Parameters
        ----------
        graph
            The graph whose area will be approximated by Riemann rectangles.
        x_range
            The minimum and maximum x-values of the rectangles. ``x_range = [x_min, x_max]``.
        dx
            The change in x-value that separates each rectangle.
        input_sample_type
            Can be any of ``"left"``, ``"right"`` or ``"center"``. Refers to where
            the sample point for the height of each Riemann Rectangle
            will be inside the segments of the partition.
        stroke_width
            The stroke_width of the border of the rectangles.
        stroke_color
            The color of the border of the rectangle.
        fill_opacity
            The opacity of the rectangles.
        color
            The colors of the rectangles. Creates a balanced gradient if multiple colors are passed.
        show_signed_area
            Indicates negative area when the curve dips below the x-axis by inverting its color.
        blend
            Sets the :attr:`stroke_color` to :attr:`fill_color`, blending the rectangles without clear separation.
        bounded_graph
            If a secondary graph is specified, encloses the area between the two curves.
        width_scale_factor
            The factor by which the width of the rectangles is scaled.

        Returns
        -------
        :class:`~.VGroup`
            A :class:`~.VGroup` containing the Riemann Rectangles.

        Examples
        --------
        .. manim:: GetRiemannRectanglesExample
            :save_last_frame:

            class GetRiemannRectanglesExample(Scene):
                def construct(self):
                    ax = Axes(y_range=[-2, 10])
                    quadratic = ax.plot(lambda x: 0.5 * x ** 2 - 0.5)

                    # the rectangles are constructed from their top right corner.
                    # passing an iterable to `color` produces a gradient
                    rects_right = ax.get_riemann_rectangles(
                        quadratic,
                        x_range=[-4, -3],
                        dx=0.25,
                        color=(TEAL, BLUE_B, DARK_BLUE),
                        input_sample_type="right",
                    )

                    # the colour of rectangles below the x-axis is inverted
                    # due to show_signed_area
                    rects_left = ax.get_riemann_rectangles(
                        quadratic, x_range=[-1.5, 1.5], dx=0.15, color=YELLOW
                    )

                    bounding_line = ax.plot(
                        lambda x: 1.5 * x, color=BLUE_B, x_range=[3.3, 6]
                    )
                    bounded_rects = ax.get_riemann_rectangles(
                        bounding_line,
                        bounded_graph=quadratic,
                        dx=0.15,
                        x_range=[4, 5],
                        show_signed_area=False,
                        color=(MAROON_A, RED_B, PURPLE_D),
                    )

                    self.add(
                        ax, bounding_line, quadratic, rects_right, rects_left, bounded_rects
                    )
        """

        # setting up x_range, overwrite user's third input
        if x_range is None:
            if bounded_graph is None:
                x_range = [graph.t_min, graph.t_max]
            else:
                x_min = max(graph.t_min, bounded_graph.t_min)
                x_max = min(graph.t_max, bounded_graph.t_max)
                x_range = [x_min, x_max]

        x_range = [*x_range[:2], dx]

        rectangles = VGroup()
        x_range = np.arange(*x_range)

        # allows passing a string to color the graph
        if type(color) is str:
            colors = [color] * len(x_range)
        else:
            colors = color_gradient(color, len(x_range))

        for x, color in zip(x_range, colors):
            if input_sample_type == "left":
                sample_input = x
            elif input_sample_type == "right":
                sample_input = x + dx
            elif input_sample_type == "center":
                sample_input = x + 0.5 * dx
            else:
                raise ValueError("Invalid input sample type")
            graph_point = self.input_to_graph_point(sample_input, graph)

            if bounded_graph is None:
                y_point = self._origin_shift(self.y_range)
            else:
                y_point = bounded_graph.underlying_function(x)

            points = VGroup(
                *list(
                    map(
                        VectorizedPoint,
                        [
                            self.coords_to_point(x, y_point),
                            self.coords_to_point(x + width_scale_factor * dx, y_point),
                            graph_point,
                        ],
                    ),
                )
            )

            rect = Rectangle().replace(points, stretch=True)
            rectangles.add(rect)

            # checks if the rectangle is under the x-axis
            if self.p2c(graph_point)[1] < y_point and show_signed_area:
                color = invert_color(color)

            # blends rectangles smoothly
            if blend:
                stroke_color = color

            rect.set_style(
                fill_color=color,
                fill_opacity=fill_opacity,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
            )

        return rectangles

    def get_area(
        self,
        graph: ParametricFunction,
        x_range: tuple[float, float] | None = None,
        color: Color | Iterable[Color] = [BLUE, GREEN],
        opacity: float = 0.3,
        bounded_graph: ParametricFunction = None,
        **kwargs,
    ):
        """Returns a :class:`~.Polygon` representing the area under the graph passed.

        Parameters
        ----------
        graph
            The graph/curve for which the area needs to be gotten.
        x_range
            The range of the minimum and maximum x-values of the area. ``x_range = [x_min, x_max]``.
        color
            The color of the area. Creates a gradient if a list of colors is provided.
        opacity
            The opacity of the area.
        bounded_graph
            If a secondary :attr:`graph` is specified, encloses the area between the two curves.
        kwargs
            Additional parameters passed to :class:`~.Polygon`.

        Returns
        -------
        :class:`~.Polygon`
            The :class:`~.Polygon` representing the area.

        Raises
        ------
        :exc:`ValueError`
            When x_ranges do not match (either area x_range, graph's x_range or bounded_graph's x_range).

        Examples
        --------
        .. manim:: GetAreaExample
            :save_last_frame:

            class GetAreaExample(Scene):
                def construct(self):
                    ax = Axes().add_coordinates()
                    curve = ax.plot(lambda x: 2 * np.sin(x), color=DARK_BLUE)
                    area = ax.get_area(
                        curve,
                        x_range=(PI / 2, 3 * PI / 2),
                        color=(GREEN_B, GREEN_D),
                        opacity=1,
                    )

                    self.add(ax, curve, area)
        """
        if x_range is None:
            a = graph.t_min
            b = graph.t_max
        else:
            a, b = x_range
        if bounded_graph is not None:
            if bounded_graph.t_min > b:
                raise ValueError(
                    f"Ranges not matching: {bounded_graph.t_min} < {b}",
                )
            if bounded_graph.t_max < a:
                raise ValueError(
                    f"Ranges not matching: {bounded_graph.t_max} > {a}",
                )
            a = max(a, bounded_graph.t_min)
            b = min(b, bounded_graph.t_max)

        if bounded_graph is None:
            points = (
                [self.c2p(a), graph.function(a)]
                + [p for p in graph.points if a <= self.p2c(p)[0] <= b]
                + [graph.function(b), self.c2p(b)]
            )
        else:
            graph_points, bounded_graph_points = (
                [g.function(a)]
                + [p for p in g.points if a <= self.p2c(p)[0] <= b]
                + [g.function(b)]
                for g in (graph, bounded_graph)
            )
            points = graph_points + bounded_graph_points[::-1]
        return Polygon(*points, **kwargs).set_opacity(opacity).set_color(color)

    def angle_of_tangent(
        self,
        x: float,
        graph: ParametricFunction,
        dx: float = 1e-8,
    ) -> float:
        """Returns the angle to the x-axis of the tangent
        to the plotted curve at a particular x-value.

        Parameters
        ----------
        x
            The x-value at which the tangent must touch the curve.
        graph
            The :class:`~.ParametricFunction` for which to calculate the tangent.
        dx
            The change in `x` used to determine the angle of the tangent to the curve.

        Returns
        -------
        :class:`float`
            The angle of the tangent to the curve.

        Examples
        --------
        .. code-block:: python

            ax = Axes()
            curve = ax.plot(lambda x: x**2)
            ax.angle_of_tangent(x=3, graph=curve)
            # 1.4056476493802699
        """

        p0 = np.array([*self.input_to_graph_coords(x, graph)])
        p1 = np.array([*self.input_to_graph_coords(x + dx, graph)])
        return angle_of_vector(p1 - p0)

    def slope_of_tangent(self, x: float, graph: ParametricFunction, **kwargs) -> float:
        """Returns the slope of the tangent to the plotted curve
        at a particular x-value.

        Parameters
        ----------
        x
            The x-value at which the tangent must touch the curve.
        graph
            The :class:`~.ParametricFunction` for which to calculate the tangent.

        Returns
        -------
        :class:`float`
            The slope of the tangent with the x axis.

        Examples
        --------
        .. code-block:: python

            ax = Axes()
            curve = ax.plot(lambda x: x**2)
            ax.slope_of_tangent(x=-2, graph=curve)
            # -3.5000000259052038
        """

        return np.tan(self.angle_of_tangent(x, graph, **kwargs))

    def plot_derivative_graph(
        self, graph: ParametricFunction, color: Color = GREEN, **kwargs
    ) -> ParametricFunction:
        """Returns the curve of the derivative of the passed graph.

        Parameters
        ----------
        graph
            The graph for which the derivative will be found.
        color
            The color of the derivative curve.
        kwargs
            Any valid keyword argument of :class:`~.ParametricFunction`.

        Returns
        -------
        :class:`~.ParametricFunction`
            The curve of the derivative.

        Examples
        --------
        .. manim:: DerivativeGraphExample
            :save_last_frame:

            class DerivativeGraphExample(Scene):
                def construct(self):
                    ax = NumberPlane(y_range=[-1, 7], background_line_style={"stroke_opacity": 0.4})

                    curve_1 = ax.plot(lambda x: x ** 2, color=PURPLE_B)
                    curve_2 = ax.plot_derivative_graph(curve_1)
                    curves = VGroup(curve_1, curve_2)

                    label_1 = ax.get_graph_label(curve_1, "x^2", x_val=-2, direction=DL)
                    label_2 = ax.get_graph_label(curve_2, "2x", x_val=3, direction=RIGHT)
                    labels = VGroup(label_1, label_2)

                    self.add(ax, curves, labels)
        """

        def deriv(x):
            return self.slope_of_tangent(x, graph)

        return self.plot(deriv, color=color, **kwargs)

    def plot_antiderivative_graph(
        self,
        graph: ParametricFunction,
        y_intercept: float = 0,
        samples: int = 50,
        use_vectorized: bool = False,
        **kwargs,
    ):
        """Plots an antiderivative graph.

        Parameters
        ----------
        graph
            The graph for which the antiderivative will be found.
        y_intercept
            The y-value at which the graph intercepts the y-axis.
        samples
            The number of points to take the area under the graph.
        use_vectorized
            Whether to use the vectorized version of the antiderivative. This means
            to pass in the generated t value array to the function. Only use this if your function supports it.
            Output should be a numpy array of shape ``[y_0, y_1, ...]``
        kwargs
            Any valid keyword argument of :class:`~.ParametricFunction`.

        Returns
        -------
        :class:`~.ParametricFunction`
            The curve of the antiderivative.


        .. note::
            This graph is plotted from the values of area under the reference graph.
            The result might not be ideal if the reference graph contains uncalculatable
            areas from x=0.

        Examples
        --------
        .. manim:: AntiderivativeExample
            :save_last_frame:

            class AntiderivativeExample(Scene):
                def construct(self):
                    ax = Axes()
                    graph1 = ax.plot(
                        lambda x: (x ** 2 - 2) / 3,
                        color=RED,
                    )
                    graph2 = ax.plot_antiderivative_graph(graph1, color=BLUE)
                    self.add(ax, graph1, graph2)
        """

        def antideriv(x):
            x_vals = np.linspace(0, x, samples, axis=1 if use_vectorized else 0)
            f_vec = np.vectorize(graph.underlying_function)
            y_vals = f_vec(x_vals)
            return np.trapz(y_vals, x_vals) + y_intercept

        return self.plot(antideriv, use_vectorized=use_vectorized, **kwargs)

    def get_secant_slope_group(
        self,
        x: float,
        graph: ParametricFunction,
        dx: float | None = None,
        dx_line_color: Color = YELLOW,
        dy_line_color: Color | None = None,
        dx_label: float | str | None = None,
        dy_label: float | str | None = None,
        include_secant_line: bool = True,
        secant_line_color: Color = GREEN,
        secant_line_length: float = 10,
    ) -> VGroup:
        """Creates two lines representing `dx` and `df`, the labels for `dx` and `df`, and
         the secant to the curve at a particular x-value.

        Parameters
        ----------
        x
            The x-value at which the secant intersects the graph for the first time.
        graph
            The curve for which the secant will be found.
        dx
            The change in `x` after which the secant exits.
        dx_line_color
            The color of the line that indicates the change in `x`.
        dy_line_color
            The color of the line that indicates the change in `y`. Defaults to the color of :attr:`graph`.
        dx_label
            The label for the `dx` line. Defaults to :class:`~.MathTex` for ``str`` and ``float`` inputs.
        dy_label
            The label for the `dy` line. Defaults to :class:`~.MathTex` for ``str`` and ``float`` inputs.
        include_secant_line
            Whether to include the secant line in the graph,
            or just the df/dx lines and labels.
        secant_line_color
            The color of the secant line.
        secant_line_length
            The length of the secant line.

        Returns
        -------
        :class:`~.VGroup`
            A group containing the elements: `dx_line`, `df_line`, and
            if applicable also :attr:`dx_label`, :attr:`df_label`, `secant_line`.

        Examples
        --------
         .. manim:: GetSecantSlopeGroupExample
            :save_last_frame:

            class GetSecantSlopeGroupExample(Scene):
                def construct(self):
                    ax = Axes(y_range=[-1, 7])
                    graph = ax.plot(lambda x: 1 / 4 * x ** 2, color=BLUE)
                    slopes = ax.get_secant_slope_group(
                        x=2.0,
                        graph=graph,
                        dx=1.0,
                        dx_label=Tex("dx = 1.0"),
                        dy_label="dy",
                        dx_line_color=GREEN_B,
                        secant_line_length=4,
                        secant_line_color=RED_D,
                    )

                    self.add(ax, graph, slopes)
        """
        group = VGroup()

        dx = dx or float(self.x_range[1] - self.x_range[0]) / 10
        dy_line_color = dy_line_color or graph.get_color()

        p1 = self.input_to_graph_point(x, graph)
        p2 = self.input_to_graph_point(x + dx, graph)
        interim_point = p2[0] * RIGHT + p1[1] * UP

        group.dx_line = Line(p1, interim_point, color=dx_line_color)
        group.df_line = Line(interim_point, p2, color=dy_line_color)
        group.add(group.dx_line, group.df_line)

        labels = VGroup()
        if dx_label is not None:
            group.dx_label = self.x_axis._create_label_tex(dx_label)
            labels.add(group.dx_label)
            group.add(group.dx_label)
        if dy_label is not None:
            group.df_label = self.x_axis._create_label_tex(dy_label)
            labels.add(group.df_label)
            group.add(group.df_label)

        if len(labels) > 0:
            max_width = 0.8 * group.dx_line.width
            max_height = 0.8 * group.df_line.height
            if labels.width > max_width:
                labels.width = max_width
            if labels.height > max_height:
                labels.height = max_height

        if dx_label is not None:
            group.dx_label.next_to(
                group.dx_line,
                np.sign(dx) * DOWN,
                buff=group.dx_label.height / 2,
            )
            group.dx_label.set_color(group.dx_line.get_color())

        if dy_label is not None:
            group.df_label.next_to(
                group.df_line,
                np.sign(dx) * RIGHT,
                buff=group.df_label.height / 2,
            )
            group.df_label.set_color(group.df_line.get_color())

        if include_secant_line:
            group.secant_line = Line(p1, p2, color=secant_line_color)
            group.secant_line.scale(
                secant_line_length / group.secant_line.get_length(),
            )
            group.add(group.secant_line)
        return group

    def get_vertical_lines_to_graph(
        self,
        graph: ParametricFunction,
        x_range: Sequence[float] | None = None,
        num_lines: int = 20,
        **kwargs,
    ) -> VGroup:
        """Obtains multiple lines from the x-axis to the curve.

        Parameters
        ----------
        graph
            The graph along which the lines are placed.
        x_range
            A list containing the lower and and upper bounds of the lines: ``x_range = [x_min, x_max]``.
        num_lines
            The number of evenly spaced lines.
        kwargs
            Additional arguments to be passed to :meth:`~.CoordinateSystem.get_vertical_line`.

        Returns
        -------
        :class:`~.VGroup`
            The :class:`~.VGroup` of the evenly spaced lines.

        Examples
        --------
        .. manim:: GetVerticalLinesToGraph
            :save_last_frame:

            class GetVerticalLinesToGraph(Scene):
                def construct(self):
                    ax = Axes(
                        x_range=[0, 8.0, 1],
                        y_range=[-1, 1, 0.2],
                        axis_config={"font_size": 24},
                    ).add_coordinates()

                    curve = ax.plot(lambda x: np.sin(x) / np.e ** 2 * x)

                    lines = ax.get_vertical_lines_to_graph(
                        curve, x_range=[0, 4], num_lines=30, color=BLUE
                    )

                    self.add(ax, curve, lines)
        """

        x_range = x_range if x_range is not None else self.x_range

        return VGroup(
            *(
                self.get_vertical_line(self.i2gp(x, graph), **kwargs)
                for x in np.linspace(x_range[0], x_range[1], num_lines)
            )
        )

    def get_T_label(
        self,
        x_val: float,
        graph: ParametricFunction,
        label: float | str | Mobject | None = None,
        label_color: Color | None = None,
        triangle_size: float = MED_SMALL_BUFF,
        triangle_color: Color | None = WHITE,
        line_func: Line = Line,
        line_color: Color = YELLOW,
    ) -> VGroup:
        """Creates a labelled triangle marker with a vertical line from the x-axis
        to a curve at a given x-value.

        Parameters
        ----------
        x_val
            The position along the curve at which the label, line and triangle will be constructed.
        graph
            The :class:`~.ParametricFunction` for which to construct the label.
        label
            The label of the vertical line and triangle.
        label_color
            The color of the label.
        triangle_size
            The size of the triangle.
        triangle_color
            The color of the triangle.
        line_func
            The function used to construct the vertical line.
        line_color
            The color of the vertical line.

        Returns
        -------
        :class:`~.VGroup`
            A :class:`~.VGroup` of the label, triangle and vertical line mobjects.

        Examples
        --------
        .. manim:: TLabelExample
            :save_last_frame:

            class TLabelExample(Scene):
                def construct(self):
                    # defines the axes and linear function
                    axes = Axes(x_range=[-1, 10], y_range=[-1, 10], x_length=9, y_length=6)
                    func = axes.plot(lambda x: x, color=BLUE)
                    # creates the T_label
                    t_label = axes.get_T_label(x_val=4, graph=func, label=Tex("x-value"))
                    self.add(axes, func, t_label)
        """

        T_label_group = VGroup()
        triangle = RegularPolygon(n=3, start_angle=np.pi / 2, stroke_width=0).set_fill(
            color=triangle_color,
            opacity=1,
        )
        triangle.height = triangle_size
        triangle.move_to(self.coords_to_point(x_val, 0), UP)
        if label is not None:
            t_label = self.x_axis._create_label_tex(label, color=label_color)
            t_label.next_to(triangle, DOWN)
            T_label_group.add(t_label)

        v_line = self.get_vertical_line(
            self.i2gp(x_val, graph),
            color=line_color,
            line_func=line_func,
        )

        T_label_group.add(triangle, v_line)

        return T_label_group


class Axes(VGroup, CoordinateSystem, metaclass=ConvertToOpenGL):
    """Creates a set of axes.

    Parameters
    ----------
    x_range
        The ``(x_min, x_max, x_step)`` values of the x-axis.
    y_range
        The ``(y_min, y_max, y_step)`` values of the y-axis.
    x_length
        The length of the x-axis.
    y_length
        The length of the y-axis.
    axis_config
        Arguments to be passed to :class:`~.NumberLine` that influences both axes.
    x_axis_config
        Arguments to be passed to :class:`~.NumberLine` that influence the x-axis.
    y_axis_config
        Arguments to be passed to :class:`~.NumberLine` that influence the y-axis.
    tips
        Whether or not to include the tips on both axes.
    kwargs
        Additional arguments to be passed to :class:`CoordinateSystem` and :class:`~.VGroup`.

    Examples
    --------
    .. manim:: LogScalingExample
        :save_last_frame:

        class LogScalingExample(Scene):
            def construct(self):
                ax = Axes(
                    x_range=[0, 10, 1],
                    y_range=[-2, 6, 1],
                    tips=False,
                    axis_config={"include_numbers": True},
                    y_axis_config={"scaling": LogBase(custom_labels=True)},
                )

                # x_min must be > 0 because log is undefined at 0.
                graph = ax.plot(lambda x: x ** 2, x_range=[0.001, 10], use_smoothing=False)
                self.add(ax, graph)
    """

    def __init__(
        self,
        x_range: Sequence[float] | None = None,
        y_range: Sequence[float] | None = None,
        x_length: float | None = round(config.frame_width) - 2,
        y_length: float | None = round(config.frame_height) - 2,
        axis_config: dict | None = None,
        x_axis_config: dict | None = None,
        y_axis_config: dict | None = None,
        tips: bool = True,
        **kwargs,
    ):
        VGroup.__init__(self, **kwargs)
        CoordinateSystem.__init__(self, x_range, y_range, x_length, y_length)

        self.axis_config = {
            "include_tip": tips,
            "numbers_to_exclude": [0],
        }
        self.x_axis_config = {}
        self.y_axis_config = {"rotation": 90 * DEGREES, "label_direction": LEFT}

        self._update_default_configs(
            (self.axis_config, self.x_axis_config, self.y_axis_config),
            (axis_config, x_axis_config, y_axis_config),
        )

        self.x_axis_config = merge_dicts_recursively(
            self.axis_config,
            self.x_axis_config,
        )
        self.y_axis_config = merge_dicts_recursively(
            self.axis_config,
            self.y_axis_config,
        )

        # excluding the origin tick removes a tick at the 0-point of the axis
        # This is desired for LinearBase because the 0 point is always the x-axis
        # For non-LinearBase, the "0-point" does not have this quality, so it must be included.

        # i.e. with LogBase range [-2, 4]:
        # it would remove the "0" tick, which is actually 10^0,
        # not the lowest tick on the graph (which is 10^-2).

        if self.x_axis_config.get("scaling") is None or isinstance(
            self.x_axis_config.get("scaling"), LinearBase
        ):
            self.x_axis_config["exclude_origin_tick"] = True
        else:
            self.x_axis_config["exclude_origin_tick"] = False

        if self.y_axis_config.get("scaling") is None or isinstance(
            self.y_axis_config.get("scaling"), LinearBase
        ):
            self.y_axis_config["exclude_origin_tick"] = True
        else:
            self.y_axis_config["exclude_origin_tick"] = False

        self.x_axis = self._create_axis(self.x_range, self.x_axis_config, self.x_length)
        self.y_axis = self._create_axis(self.y_range, self.y_axis_config, self.y_length)

        # Add as a separate group in case various other
        # mobjects are added to self, as for example in
        # NumberPlane below
        self.axes = VGroup(self.x_axis, self.y_axis)
        self.add(*self.axes)

        # finds the middle-point on each axis
        lines_center_point = [
            axis.scaling.function((axis.x_range[1] + axis.x_range[0]) / 2)
            for axis in self.axes
        ]

        self.shift(-self.coords_to_point(*lines_center_point))

    @staticmethod
    def _update_default_configs(
        default_configs: tuple[dict[Any, Any]], passed_configs: tuple[dict[Any, Any]]
    ):
        """Takes in two tuples of dicts and return modifies the first such that values from
        ``passed_configs`` overwrite values in ``default_configs``. If a key does not exist
        in default_configs, it is added to the dict.

        This method is useful for having defaults in a class and being able to overwrite
        them with user-defined input.

        Parameters
        ----------
        default_configs
            The dict that will be updated.
        passed_configs
            The dict that will be used to update.

        To create a tuple with one dictionary, add a comma after the element:

        .. code-block:: python

            self._update_default_configs(
                (dict_1,)(
                    dict_2,
                )
            )
        """

        for default_config, passed_config in zip(default_configs, passed_configs):
            if passed_config is not None:
                update_dict_recursively(default_config, passed_config)

    def _create_axis(
        self,
        range_terms: Sequence[float],
        axis_config: dict,
        length: float,
    ) -> NumberLine:
        """Creates an axis and dynamically adjusts its position depending on where 0 is located on the line.

        Parameters
        ----------
        range_terms
            The range of the the axis : ``(x_min, x_max, x_step)``.
        axis_config
            Additional parameters that are passed to :class:`~.NumberLine`.
        length
            The length of the axis.

        Returns
        -------
        :class:`NumberLine`
            Returns a number line based on ``range_terms``.
        """
        axis_config["length"] = length
        axis = NumberLine(range_terms, **axis_config)

        # without the call to _origin_shift, graph does not exist when min > 0 or max < 0
        # shifts the axis so that 0 is centered
        axis.shift(-axis.number_to_point(self._origin_shift([axis.x_min, axis.x_max])))
        return axis

    def coords_to_point(
        self, *coords: Sequence[float] | Sequence[Sequence[float]] | np.ndarray
    ) -> np.ndarray:
        """Accepts coordinates from the axes and returns a point with respect to the scene.

        Parameters
        ----------
        coords
            The coordinates. Each coord is passed as a separate argument: ``ax.coords_to_point(1, 2, 3)``.

            Also accepts a list of coordinates

            ``ax.coords_to_point( [x_0, x_1, ...], [y_0, y_1, ...], ... )``

            ``ax.coords_to_point( [[x_0, y_0, z_0], [x_1, y_1, z_1]] )``

        Returns
        -------
        np.ndarray
            A point with respect to the scene's coordinate system.
            The shape of the array will be similar to the shape of the input.

        Examples
        --------

        .. code-block:: pycon

            >>> from manim import Axes
            >>> import numpy as np
            >>> ax = Axes()
            >>> np.around(ax.coords_to_point(1, 0, 0), 2)
            array([0.86, 0.  , 0.  ])
            >>> np.around(ax.coords_to_point([[0, 1], [1, 1], [1, 0]]), 2)
            array([[0.  , 0.75, 0.  ],
                   [0.86, 0.75, 0.  ],
                   [0.86, 0.  , 0.  ]])
            >>> np.around(
            ...     ax.coords_to_point([0, 1, 1], [1, 1, 0]), 2
            ... )  # Transposed version of the above
            array([[0.  , 0.86, 0.86],
                   [0.75, 0.75, 0.  ],
                   [0.  , 0.  , 0.  ]])

        .. manim:: CoordsToPointExample
            :save_last_frame:

            class CoordsToPointExample(Scene):
                def construct(self):
                    ax = Axes().add_coordinates()

                    # a dot with respect to the axes
                    dot_axes = Dot(ax.coords_to_point(2, 2), color=GREEN)
                    lines = ax.get_lines_to_point(ax.c2p(2,2))

                    # a dot with respect to the scene
                    # the default plane corresponds to the coordinates of the scene.
                    plane = NumberPlane()
                    dot_scene = Dot((2,2,0), color=RED)

                    self.add(plane, dot_scene, ax, dot_axes, lines)
        """
        origin = self.x_axis.number_to_point(
            self._origin_shift([self.x_axis.x_min, self.x_axis.x_max]),
        )

        coords = np.asarray(coords)

        # if called like coords_to_point(1, 2, 3), then coords is a 1x3 array
        transposed = False
        if coords.ndim == 1:
            # original implementation of coords_to_point for performance in the legacy case
            result = np.array(origin)
            for axis, number in zip(self.get_axes(), coords):
                result += axis.number_to_point(number) - origin
            return result
        # if called like coords_to_point([1, 2, 3],[4, 5, 6]), then it shall be used as [1,4], [2,5], [3,6] and return the points as ([x_0,x_1],[y_0,y_1],[z_0,z_1])
        elif coords.ndim == 2:
            coords = coords.T
            transposed = True
        # if called like coords_to_point(np.array([[1, 2, 3],[4,5,6]])), reduce dimension by 1
        elif coords.ndim == 3:
            coords = np.squeeze(coords)
        # else the coords is a Nx1, Nx2, Nx3 array so we do not need to modify the array

        points = origin + np.sum(
            [
                axis.number_to_point(number) - origin
                for number, axis in zip(coords.T, self.get_axes())
            ],
            axis=0,
        )
        # if called with single coord, then return a point instead of a list of points
        if transposed:
            return points.T
        return points

    def point_to_coords(self, point: Sequence[float]) -> np.ndarray:
        """Accepts a point from the scene and returns its coordinates with respect to the axes.

        Parameters
        ----------
        point
            The point, i.e. ``RIGHT`` or ``[0, 1, 0]``.
            Also accepts a list of points as ``[RIGHT, [0, 1, 0]]``.

        Returns
        -------
        np.ndarray[float]
            The coordinates on the axes, i.e. ``[4.0, 7.0]``.
            Or a list of coordinates if `point` is a list of points.

        Examples
        --------

        .. code-block:: pycon

            >>> from manim import Axes, RIGHT
            >>> import numpy as np
            >>> ax = Axes(x_range=[0, 10, 2])
            >>> np.around(ax.point_to_coords(RIGHT), 2)
            array([5.83, 0.  ])
            >>> np.around(ax.point_to_coords([[0, 0, 1], [1, 0, 0]]), 2)
            array([[5.  , 0.  ],
                   [5.83, 0.  ]])


        .. manim:: PointToCoordsExample
            :save_last_frame:

            class PointToCoordsExample(Scene):
                def construct(self):
                    ax = Axes(x_range=[0, 10, 2]).add_coordinates()
                    circ = Circle(radius=0.5).shift(UR * 2)

                    # get the coordinates of the circle with respect to the axes
                    coords = np.around(ax.point_to_coords(circ.get_right()), decimals=2)

                    label = (
                        Matrix([[coords[0]], [coords[1]]]).scale(0.75).next_to(circ, RIGHT)
                    )

                    self.add(ax, circ, label, Dot(circ.get_right()))
        """
        point = np.asarray(point)
        result = np.asarray([axis.point_to_number(point) for axis in self.get_axes()])
        if point.ndim == 2:
            return result.T
        return result

    def get_axes(self) -> VGroup:
        """Gets the axes.

        Returns
        -------
        :class:`~.VGroup`
            A pair of axes.
        """
        return self.axes

    def plot_line_graph(
        self,
        x_values: Iterable[float],
        y_values: Iterable[float],
        z_values: Iterable[float] | None = None,
        line_color: Color = YELLOW,
        add_vertex_dots: bool = True,
        vertex_dot_radius: float = DEFAULT_DOT_RADIUS,
        vertex_dot_style: dict | None = None,
        **kwargs,
    ) -> VDict:
        """Draws a line graph.

        The graph connects the vertices formed from zipping
        ``x_values``, ``y_values`` and ``z_values``. Also adds :class:`Dots <.Dot>` at the
        vertices if ``add_vertex_dots`` is set to ``True``.

        Parameters
        ----------
        x_values
            Iterable of values along the x-axis.
        y_values
            Iterable of values along the y-axis.
        z_values
            Iterable of values (zeros if z_values is None) along the z-axis.
        line_color
            Color for the line graph.
        add_vertex_dots
            Whether or not to add :class:`~.Dot` at each vertex.
        vertex_dot_radius
            Radius for the :class:`~.Dot` at each vertex.
        vertex_dot_style
            Style arguments to be passed into :class:`~.Dot` at each vertex.
        kwargs
            Additional arguments to be passed into :class:`~.VMobject`.

        Returns
        -------
        :class:`~.VDict`
            A VDict containing both the line and dots (if specified). The line can be accessed with: ``line_graph["line_graph"]``.
            The dots can be accessed with: ``line_graph["vertex_dots"]``.

        Examples
        --------
        .. manim:: LineGraphExample
            :save_last_frame:

            class LineGraphExample(Scene):
                def construct(self):
                    plane = NumberPlane(
                        x_range = (0, 7),
                        y_range = (0, 5),
                        x_length = 7,
                        axis_config={"include_numbers": True},
                    )
                    plane.center()
                    line_graph = plane.plot_line_graph(
                        x_values = [0, 1.5, 2, 2.8, 4, 6.25],
                        y_values = [1, 3, 2.25, 4, 2.5, 1.75],
                        line_color=GOLD_E,
                        vertex_dot_style=dict(stroke_width=3,  fill_color=PURPLE),
                        stroke_width = 4,
                    )
                    self.add(plane, line_graph)
        """
        x_values, y_values = map(np.array, (x_values, y_values))
        if z_values is None:
            z_values = np.zeros(x_values.shape)

        line_graph = VDict()
        graph = VGroup(color=line_color, **kwargs)

        vertices = [
            self.coords_to_point(x, y, z)
            for x, y, z in zip(x_values, y_values, z_values)
        ]
        graph.set_points_as_corners(vertices)
        line_graph["line_graph"] = graph

        if add_vertex_dots:
            vertex_dot_style = vertex_dot_style or {}
            vertex_dots = VGroup(
                *(
                    Dot(point=vertex, radius=vertex_dot_radius, **vertex_dot_style)
                    for vertex in vertices
                )
            )
            line_graph["vertex_dots"] = vertex_dots

        return line_graph

    @staticmethod
    def _origin_shift(axis_range: Sequence[float]) -> float:
        """Determines how to shift graph mobjects to compensate when 0 is not on the axis.

        Parameters
        ----------
        axis_range
            The range of the axis : ``(x_min, x_max, x_step)``.
        """
        if axis_range[0] > 0:
            # min greater than 0
            return axis_range[0]
        if axis_range[1] < 0:
            # max less than 0
            return axis_range[1]
        else:
            return 0


class ThreeDAxes(Axes):
    """A 3-dimensional set of axes.

    Parameters
    ----------
    x_range
        The ``[x_min, x_max, x_step]`` values of the x-axis.
    y_range
        The ``[y_min, y_max, y_step]`` values of the y-axis.
    z_range
        The ``[z_min, z_max, z_step]`` values of the z-axis.
    x_length
        The length of the x-axis.
    y_length
        The length of the y-axis.
    z_length
        The length of the z-axis.
    z_axis_config
        Arguments to be passed to :class:`~.NumberLine` that influence the z-axis.
    z_normal
        The direction of the normal.
    num_axis_pieces
        The number of pieces used to construct the axes.
    light_source
        The direction of the light source.
    depth
        Currently non-functional.
    gloss
        Currently non-functional.
    kwargs
        Additional arguments to be passed to :class:`Axes`.
    """

    def __init__(
        self,
        x_range: Sequence[float] | None = (-6, 6, 1),
        y_range: Sequence[float] | None = (-5, 5, 1),
        z_range: Sequence[float] | None = (-4, 4, 1),
        x_length: float | None = config.frame_height + 2.5,
        y_length: float | None = config.frame_height + 2.5,
        z_length: float | None = config.frame_height - 1.5,
        z_axis_config: dict | None = None,
        z_normal: Sequence[float] = DOWN,
        num_axis_pieces: int = 20,
        light_source: Sequence[float] = 9 * DOWN + 7 * LEFT + 10 * OUT,
        # opengl stuff (?)
        depth=None,
        gloss=0.5,
        **kwargs,
    ):

        super().__init__(
            x_range=x_range,
            x_length=x_length,
            y_range=y_range,
            y_length=y_length,
            **kwargs,
        )

        self.z_range = z_range
        self.z_length = z_length

        self.z_axis_config = {}
        self._update_default_configs((self.z_axis_config,), (z_axis_config,))
        self.z_axis_config = merge_dicts_recursively(
            self.axis_config,
            self.z_axis_config,
        )

        self.z_normal = z_normal
        self.num_axis_pieces = num_axis_pieces

        self.light_source = light_source

        self.dimension = 3

        if self.z_axis_config.get("scaling") is None or isinstance(
            self.z_axis_config.get("scaling"), LinearBase
        ):
            self.z_axis_config["exclude_origin_tick"] = True
        else:
            self.z_axis_config["exclude_origin_tick"] = False

        z_axis = self._create_axis(self.z_range, self.z_axis_config, self.z_length)

        # [ax.x_min, ax.x_max] used to account for LogBase() scaling
        # where ax.x_range[0] != ax.x_min
        z_origin = self._origin_shift([z_axis.x_min, z_axis.x_max])

        z_axis.rotate_about_number(z_origin, -PI / 2, UP)
        z_axis.rotate_about_number(z_origin, angle_of_vector(self.z_normal))
        z_axis.shift(-z_axis.number_to_point(z_origin))
        z_axis.shift(
            self.x_axis.number_to_point(
                self._origin_shift([self.x_axis.x_min, self.x_axis.x_max]),
            ),
        )

        self.axes.add(z_axis)
        self.add(z_axis)
        self.z_axis = z_axis

        if config.renderer != "opengl":
            self._add_3d_pieces()
            self._set_axis_shading()

    def _add_3d_pieces(self):
        for axis in self.axes:
            axis.pieces = VGroup(*axis.get_pieces(self.num_axis_pieces))
            axis.add(axis.pieces)
            axis.set_stroke(width=0, family=False)
            axis.set_shade_in_3d(True)

    def _set_axis_shading(self):
        def make_func(axis):
            vect = self.light_source
            return lambda: (
                axis.get_edge_center(-vect),
                axis.get_edge_center(vect),
            )

        for axis in self:
            for submob in axis.family_members_with_points():
                submob.get_gradient_start_and_end_points = make_func(axis)
                submob.get_unit_normal = lambda a: np.ones(3)
                submob.set_sheen(0.2)

    def get_z_axis_label(
        self,
        label: float | str | Mobject,
        edge: Sequence[float] = OUT,
        direction: Sequence[float] = RIGHT,
        buff: float = SMALL_BUFF,
        rotation=PI / 2,
        rotation_axis=RIGHT,
        **kwargs,
    ) -> Mobject:
        """Generate a z-axis label.

        Parameters
        ----------
        label
            The label. Defaults to :class:`~.MathTex` for ``str`` and ``float`` inputs.
        edge
            The edge of the x-axis to which the label will be added, by default ``UR``.
        direction
            Allows for further positioning of the label from an edge, by default ``UR``.
        buff
            The distance of the label from the line.
        rotation
            The angle at which to rotate the label, by default ``PI/2``.
        rotation_axis
            The axis about which to rotate the label, by default ``RIGHT``.

        Returns
        -------
        :class:`~.Mobject`
            The positioned label.

        Examples
        --------
        .. manim:: GetZAxisLabelExample
            :save_last_frame:

            class GetZAxisLabelExample(ThreeDScene):
                def construct(self):
                    ax = ThreeDAxes()
                    lab = ax.get_z_axis_label(Tex("$z$-label"))
                    self.set_camera_orientation(phi=2*PI/5, theta=PI/5)
                    self.add(ax, lab)
        """

        positioned_label = self._get_axis_label(
            label, self.get_z_axis(), edge, direction, buff=buff, **kwargs
        )
        positioned_label.rotate(rotation, axis=rotation_axis)
        return positioned_label


class NumberPlane(Axes):
    """Creates a cartesian plane with background lines.

    Parameters
    ----------
    x_range
        The ``[x_min, x_max, x_step]`` values of the plane in the horizontal direction.
    y_range
        The ``[y_min, y_max, y_step]`` values of the plane in the vertical direction.
    x_length
        The width of the plane.
    y_length
        The height of the plane.
    background_line_style
        Arguments that influence the construction of the background lines of the plane.
    faded_line_style
        Similar to :attr:`background_line_style`, affects the construction of the scene's background lines.
    faded_line_ratio
        Determines the number of boxes within the background lines: :code:`2` = 4 boxes, :code:`3` = 9 boxes.
    make_smooth_after_applying_functions
        Currently non-functional.
    kwargs
        Additional arguments to be passed to :class:`Axes`.


    .. note::
        If :attr:`x_length` or :attr:`y_length` are not defined, they are automatically calculated such that
        one unit on each axis is one Manim unit long.

    Examples
    --------
    .. manim:: NumberPlaneExample
        :save_last_frame:

        class NumberPlaneExample(Scene):
            def construct(self):
                number_plane = NumberPlane(
                    background_line_style={
                        "stroke_color": TEAL,
                        "stroke_width": 4,
                        "stroke_opacity": 0.6
                    }
                )
                self.add(number_plane)

    .. manim:: NumberPlaneScaled
        :save_last_frame:

        class NumberPlaneScaled(Scene):
            def construct(self):
                number_plane = NumberPlane(
                    x_range=(-4, 11, 1),
                    y_range=(-3, 3, 1),
                    x_length=5,
                    y_length=2,
                ).move_to(LEFT*3)

                number_plane_scaled_y = NumberPlane(
                    x_range=(-4, 11, 1),
                    x_length=5,
                    y_length=4,
                ).move_to(RIGHT*3)

                self.add(number_plane)
                self.add(number_plane_scaled_y)
    """

    def __init__(
        self,
        x_range: Sequence[float]
        | None = (
            -config["frame_x_radius"],
            config["frame_x_radius"],
            1,
        ),
        y_range: Sequence[float]
        | None = (
            -config["frame_y_radius"],
            config["frame_y_radius"],
            1,
        ),
        x_length: float | None = None,
        y_length: float | None = None,
        background_line_style: dict | None = None,
        faded_line_style: dict | None = None,
        faded_line_ratio: int = 1,
        make_smooth_after_applying_functions: bool = True,
        **kwargs,
    ):

        # configs
        self.axis_config = {
            "stroke_width": 2,
            "include_ticks": False,
            "include_tip": False,
            "line_to_number_buff": SMALL_BUFF,
            "label_direction": DR,
            "font_size": 24,
        }
        self.y_axis_config = {"label_direction": DR}
        self.background_line_style = {
            "stroke_color": BLUE_D,
            "stroke_width": 2,
            "stroke_opacity": 1,
        }

        self._update_default_configs(
            (self.axis_config, self.y_axis_config, self.background_line_style),
            (
                kwargs.pop("axis_config", None),
                kwargs.pop("y_axis_config", None),
                background_line_style,
            ),
        )

        # Defaults to a faded version of line_config
        self.faded_line_style = faded_line_style
        self.faded_line_ratio = faded_line_ratio
        self.make_smooth_after_applying_functions = make_smooth_after_applying_functions

        # init
        super().__init__(
            x_range=x_range,
            y_range=y_range,
            x_length=x_length,
            y_length=y_length,
            axis_config=self.axis_config,
            y_axis_config=self.y_axis_config,
            **kwargs,
        )

        self._init_background_lines()

    def _init_background_lines(self):
        """Will init all the lines of NumberPlanes (faded or not)"""
        if self.faded_line_style is None:
            style = dict(self.background_line_style)
            # For anything numerical, like stroke_width
            # and stroke_opacity, chop it in half
            for key in style:
                if isinstance(style[key], numbers.Number):
                    style[key] *= 0.5
            self.faded_line_style = style

        self.background_lines, self.faded_lines = self._get_lines()

        self.background_lines.set_style(
            **self.background_line_style,
        )
        self.faded_lines.set_style(
            **self.faded_line_style,
        )
        self.add_to_back(
            self.faded_lines,
            self.background_lines,
        )

    def _get_lines(self) -> tuple[VGroup, VGroup]:
        """Generate all the lines, faded and not faded.
         Two sets of lines are generated: one parallel to the X-axis, and parallel to the Y-axis.

        Returns
        -------
        Tuple[:class:`~.VGroup`, :class:`~.VGroup`]
            The first (i.e the non faded lines) and second (i.e the faded lines) sets of lines, respectively.
        """
        x_axis = self.get_x_axis()
        y_axis = self.get_y_axis()

        x_lines1, x_lines2 = self._get_lines_parallel_to_axis(
            x_axis,
            y_axis,
            self.y_axis.x_range[2],
            self.faded_line_ratio,
        )

        y_lines1, y_lines2 = self._get_lines_parallel_to_axis(
            y_axis,
            x_axis,
            self.x_axis.x_range[2],
            self.faded_line_ratio,
        )

        # TODO this was added so that we can run tests on NumberPlane
        # In the future these attributes will be tacked onto self.background_lines
        self.x_lines = x_lines1
        self.y_lines = y_lines1
        lines1 = VGroup(*x_lines1, *y_lines1)
        lines2 = VGroup(*x_lines2, *y_lines2)

        return lines1, lines2

    def _get_lines_parallel_to_axis(
        self,
        axis_parallel_to: NumberLine,
        axis_perpendicular_to: NumberLine,
        freq: float,
        ratio_faded_lines: int,
    ) -> tuple[VGroup, VGroup]:
        """Generate a set of lines parallel to an axis.

        Parameters
        ----------
        axis_parallel_to
            The axis with which the lines will be parallel.
        axis_perpendicular_to
            The axis with which the lines will be perpendicular.
        ratio_faded_lines
            The ratio between the space between faded lines and the space between non-faded lines.
        freq
            Frequency of non-faded lines (number of non-faded lines per graph unit).

        Returns
        -------
        Tuple[:class:`~.VGroup`, :class:`~.VGroup`]
            The first (i.e the non-faded lines parallel to `axis_parallel_to`) and second
             (i.e the faded lines parallel to `axis_parallel_to`) sets of lines, respectively.
        """

        line = Line(axis_parallel_to.get_start(), axis_parallel_to.get_end())
        if ratio_faded_lines == 0:  # don't show faded lines
            ratio_faded_lines = 1  # i.e. set ratio to 1
        step = (1 / ratio_faded_lines) * freq
        lines1 = VGroup()
        lines2 = VGroup()
        unit_vector_axis_perp_to = axis_perpendicular_to.get_unit_vector()

        # need to unpack all three values
        x_min, x_max, _ = axis_perpendicular_to.x_range

        # account for different axis scalings (logarithmic), where
        # negative values do not exist and [-2 , 4] should output lines
        # similar to [0, 6]
        if axis_perpendicular_to.x_min > 0 and x_min < 0:
            x_min, x_max = (0, np.abs(x_min) + np.abs(x_max))

        # min/max used in case range does not include 0. i.e. if (2,6):
        # the range becomes (0,4), not (0,6).
        ranges = (
            [0],
            np.arange(step, min(x_max - x_min, x_max), step),
            np.arange(-step, max(x_min - x_max, x_min), -step),
        )

        for inputs in ranges:
            for k, x in enumerate(inputs):
                new_line = line.copy()
                new_line.shift(unit_vector_axis_perp_to * x)
                if k % ratio_faded_lines == 0:
                    lines1.add(new_line)
                else:
                    lines2.add(new_line)
        return lines1, lines2

    def get_vector(self, coords: Sequence[float], **kwargs):
        kwargs["buff"] = 0
        return Arrow(
            self.coords_to_point(0, 0), self.coords_to_point(*coords), **kwargs
        )

    def prepare_for_nonlinear_transform(self, num_inserted_curves: int = 50):
        for mob in self.family_members_with_points():
            num_curves = mob.get_num_curves()
            if num_inserted_curves > num_curves:
                mob.insert_n_curves(num_inserted_curves - num_curves)
        return self


class PolarPlane(Axes):
    r"""Creates a polar plane with background lines.

    Parameters
    ----------
    azimuth_step
        The number of divisions in the azimuth (also known as the `angular coordinate` or `polar angle`). If ``None`` is specified then it will use the default
        specified by ``azimuth_units``:

        - ``"PI radians"`` or ``"TAU radians"``: 20
        - ``"degrees"``: 36
        - ``"gradians"``: 40
        - ``None``: 1

        A non-integer value will result in a partial division at the end of the circle.

    size
        The diameter of the plane.

    radius_step
        The distance between faded radius lines.

    radius_max
        The maximum value of the radius.

    azimuth_units
        Specifies a default labelling system for the azimuth. Choices are:

        - ``"PI radians"``: Fractional labels in the interval :math:`\left[0, 2\pi\right]` with :math:`\pi` as a constant.
        - ``"TAU radians"``: Fractional labels in the interval :math:`\left[0, \tau\right]` (where :math:`\tau = 2\pi`) with :math:`\tau` as a constant.
        - ``"degrees"``: Decimal labels in the interval :math:`\left[0, 360\right]` with a degree (:math:`^{\circ}`) symbol.
        - ``"gradians"``: Decimal labels in the interval :math:`\left[0, 400\right]` with a superscript "g" (:math:`^{g}`).
        - ``None``: Decimal labels in the interval :math:`\left[0, 1\right]`.

    azimuth_compact_fraction
        If the ``azimuth_units`` choice has fractional labels, choose whether to
        combine the constant in a compact form :math:`\tfrac{xu}{y}` as opposed to
        :math:`\tfrac{x}{y}u`, where :math:`u` is the constant.

    azimuth_offset
        The angle offset of the azimuth, expressed in radians.

    azimuth_direction
        The direction of the azimuth.

        - ``"CW"``: Clockwise.
        - ``"CCW"``: Anti-clockwise.

    azimuth_label_buff
        The buffer for the azimuth labels.

    azimuth_label_font_size
        The font size of the azimuth labels.

    radius_config
        The axis config for the radius.

    Examples
    --------
    .. manim:: PolarPlaneExample
        :ref_classes: PolarPlane
        :save_last_frame:

        class PolarPlaneExample(Scene):
            def construct(self):
                polarplane_pi = PolarPlane(
                    azimuth_units="PI radians",
                    size=6,
                    azimuth_label_font_size=33.6,
                    radius_config={"font_size": 33.6},
                ).add_coordinates()
                self.add(polarplane_pi)
    """

    def __init__(
        self,
        radius_max: float = config["frame_y_radius"],
        size: float | None = None,
        radius_step: float = 1,
        azimuth_step: float | None = None,
        azimuth_units: str | None = "PI radians",
        azimuth_compact_fraction: bool = True,
        azimuth_offset: float = 0,
        azimuth_direction: str = "CCW",
        azimuth_label_buff: float = SMALL_BUFF,
        azimuth_label_font_size: float = 24,
        radius_config: dict | None = None,
        background_line_style: dict | None = None,
        faded_line_style: dict | None = None,
        faded_line_ratio: int = 1,
        make_smooth_after_applying_functions: bool = True,
        **kwargs,
    ):
        # error catching
        if azimuth_units in ["PI radians", "TAU radians", "degrees", "gradians", None]:
            self.azimuth_units = azimuth_units
        else:
            raise ValueError(
                "Invalid azimuth units. Expected one of: PI radians, TAU radians, degrees, gradians or None.",
            )

        if azimuth_direction in ["CW", "CCW"]:
            self.azimuth_direction = azimuth_direction
        else:
            raise ValueError("Invalid azimuth units. Expected one of: CW, CCW.")

        # configs
        self.radius_config = {
            "stroke_width": 2,
            "include_ticks": False,
            "include_tip": False,
            "line_to_number_buff": SMALL_BUFF,
            "label_direction": DL,
            "font_size": 24,
        }

        self.background_line_style = {
            "stroke_color": BLUE_D,
            "stroke_width": 2,
            "stroke_opacity": 1,
        }

        self.azimuth_step = (
            (
                {
                    "PI radians": 20,
                    "TAU radians": 20,
                    "degrees": 36,
                    "gradians": 40,
                    None: 1,
                }[azimuth_units]
            )
            if azimuth_step is None
            else azimuth_step
        )

        self._update_default_configs(
            (self.radius_config, self.background_line_style),
            (radius_config, background_line_style),
        )

        # Defaults to a faded version of line_config
        self.faded_line_style = faded_line_style
        self.faded_line_ratio = faded_line_ratio
        self.make_smooth_after_applying_functions = make_smooth_after_applying_functions
        self.azimuth_offset = azimuth_offset
        self.azimuth_label_buff = azimuth_label_buff
        self.azimuth_label_font_size = azimuth_label_font_size
        self.azimuth_compact_fraction = azimuth_compact_fraction

        # init

        super().__init__(
            x_range=np.array((-radius_max, radius_max, radius_step)),
            y_range=np.array((-radius_max, radius_max, radius_step)),
            x_length=size,
            y_length=size,
            axis_config=self.radius_config,
            **kwargs,
        )

        self._init_background_lines()

    def _init_background_lines(self):
        """Will init all the lines of NumberPlanes (faded or not)"""
        if self.faded_line_style is None:
            style = dict(self.background_line_style)
            # For anything numerical, like stroke_width
            # and stroke_opacity, chop it in half
            for key in style:
                if isinstance(style[key], numbers.Number):
                    style[key] *= 0.5
            self.faded_line_style = style

        self.background_lines, self.faded_lines = self._get_lines()
        self.background_lines.set_style(
            **self.background_line_style,
        )
        self.faded_lines.set_style(
            **self.faded_line_style,
        )
        self.add_to_back(
            self.faded_lines,
            self.background_lines,
        )

    def _get_lines(self) -> tuple[VGroup, VGroup]:
        """Generate all the lines and circles, faded and not faded.

        Returns
        -------
        Tuple[:class:`~.VGroup`, :class:`~.VGroup`]
            The first (i.e the non faded lines and circles) and second (i.e the faded lines and circles) sets of lines and circles, respectively.
        """
        center = self.get_origin()
        ratio_faded_lines = self.faded_line_ratio
        offset = self.azimuth_offset

        if ratio_faded_lines == 0:  # don't show faded lines
            ratio_faded_lines = 1  # i.e. set ratio to 1
        rstep = (1 / ratio_faded_lines) * self.x_axis.x_range[2]
        astep = (1 / ratio_faded_lines) * (TAU * (1 / self.azimuth_step))
        rlines1 = VGroup()
        rlines2 = VGroup()
        alines1 = VGroup()
        alines2 = VGroup()

        rinput = np.arange(0, self.x_axis.x_range[1] + rstep, rstep)
        ainput = np.arange(0, TAU, astep)

        unit_vector = self.x_axis.get_unit_vector()[0]

        for k, x in enumerate(rinput):
            new_line = Circle(radius=x * unit_vector)
            if k % ratio_faded_lines == 0:
                alines1.add(new_line)
            else:
                alines2.add(new_line)

        line = Line(center, self.get_x_axis().get_end())

        for k, x in enumerate(ainput):
            new_line = line.copy()
            new_line.rotate(x + offset, about_point=center)
            if k % ratio_faded_lines == 0:
                rlines1.add(new_line)
            else:
                rlines2.add(new_line)

        lines1 = VGroup(*rlines1, *alines1)
        lines2 = VGroup(*rlines2, *alines2)
        return lines1, lines2

    def get_axes(self) -> VGroup:
        """Gets the axes.

        Returns
        -------
        :class:`~.VGroup`
            A pair of axes.
        """
        return self.axes

    def get_vector(self, coords, **kwargs):
        kwargs["buff"] = 0
        return Arrow(
            self.coords_to_point(0, 0), self.coords_to_point(*coords), **kwargs
        )

    def prepare_for_nonlinear_transform(self, num_inserted_curves=50):
        for mob in self.family_members_with_points():
            num_curves = mob.get_num_curves()
            if num_inserted_curves > num_curves:
                mob.insert_n_curves(num_inserted_curves - num_curves)
        return self

    def get_coordinate_labels(
        self,
        r_values: Iterable[float] | None = None,
        a_values: Iterable[float] | None = None,
        **kwargs,
    ) -> VDict:
        """Gets labels for the coordinates

        Parameters
        ----------
        r_values
            Iterable of values along the radius, by default None.
        a_values
            Iterable of values along the azimuth, by default None.

        Returns
        -------
        VDict
            Labels for the radius and azimuth values.
        """
        if r_values is None:
            r_values = [r for r in self.get_x_axis().get_tick_range() if r >= 0]
        if a_values is None:
            a_values = np.arange(0, 1, 1 / self.azimuth_step)
        r_mobs = self.get_x_axis().add_numbers(r_values)
        if self.azimuth_direction == "CCW":
            d = 1
        elif self.azimuth_direction == "CW":
            d = -1
        else:
            raise ValueError("Invalid azimuth direction. Expected one of: CW, CCW")
        a_points = [
            {
                "label": i,
                "point": np.array(
                    [
                        self.get_right()[0]
                        * np.cos(d * (i * TAU) + self.azimuth_offset),
                        self.get_right()[0]
                        * np.sin(d * (i * TAU) + self.azimuth_offset),
                        0,
                    ],
                ),
            }
            for i in a_values
        ]
        if self.azimuth_units == "PI radians" or self.azimuth_units == "TAU radians":
            a_tex = [
                self.get_radian_label(
                    i["label"],
                    font_size=self.azimuth_label_font_size,
                ).next_to(
                    i["point"],
                    direction=i["point"],
                    aligned_edge=i["point"],
                    buff=self.azimuth_label_buff,
                )
                for i in a_points
            ]
        elif self.azimuth_units == "degrees":
            a_tex = [
                MathTex(
                    f'{360 * i["label"]:g}' + r"^{\circ}",
                    font_size=self.azimuth_label_font_size,
                ).next_to(
                    i["point"],
                    direction=i["point"],
                    aligned_edge=i["point"],
                    buff=self.azimuth_label_buff,
                )
                for i in a_points
            ]
        elif self.azimuth_units == "gradians":
            a_tex = [
                MathTex(
                    f'{400 * i["label"]:g}' + r"^{g}",
                    font_size=self.azimuth_label_font_size,
                ).next_to(
                    i["point"],
                    direction=i["point"],
                    aligned_edge=i["point"],
                    buff=self.azimuth_label_buff,
                )
                for i in a_points
            ]
        elif self.azimuth_units is None:
            a_tex = [
                MathTex(
                    f'{i["label"]:g}',
                    font_size=self.azimuth_label_font_size,
                ).next_to(
                    i["point"],
                    direction=i["point"],
                    aligned_edge=i["point"],
                    buff=self.azimuth_label_buff,
                )
                for i in a_points
            ]
        a_mobs = VGroup(*a_tex)
        self.coordinate_labels = VGroup(r_mobs, a_mobs)
        return self.coordinate_labels

    def add_coordinates(
        self,
        r_values: Iterable[float] | None = None,
        a_values: Iterable[float] | None = None,
    ):
        """Adds the coordinates.

        Parameters
        ----------
        r_values
            Iterable of values along the radius, by default None.
        a_values
            Iterable of values along the azimuth, by default None.
        """
        self.add(self.get_coordinate_labels(r_values, a_values))
        return self

    def get_radian_label(self, number, font_size=24, **kwargs):
        constant_label = {"PI radians": r"\pi", "TAU radians": r"\tau"}[
            self.azimuth_units
        ]
        division = number * {"PI radians": 2, "TAU radians": 1}[self.azimuth_units]
        frac = fr.Fraction(division).limit_denominator(max_denominator=100)
        if frac.numerator == 0 & frac.denominator == 0:
            string = r"0"
        elif frac.numerator == 1 and frac.denominator == 1:
            string = constant_label
        elif frac.numerator == 1:
            if self.azimuth_compact_fraction:
                string = (
                    r"\tfrac{" + constant_label + r"}{" + str(frac.denominator) + "}"
                )
            else:
                string = r"\tfrac{1}{" + str(frac.denominator) + "}" + constant_label
        elif frac.denominator == 1:
            string = str(frac.numerator) + constant_label

        else:
            if self.azimuth_compact_fraction:
                string = (
                    r"\tfrac{"
                    + str(frac.numerator)
                    + constant_label
                    + r"}{"
                    + str(frac.denominator)
                    + r"}"
                )
            else:
                string = (
                    r"\tfrac{"
                    + str(frac.numerator)
                    + r"}{"
                    + str(frac.denominator)
                    + r"}"
                    + constant_label
                )

        return MathTex(string, font_size=font_size, **kwargs)


class ComplexPlane(NumberPlane):
    """A :class:`~.NumberPlane` specialized for use with complex numbers.

    Examples
    --------
    .. manim:: ComplexPlaneExample
        :save_last_frame:
        :ref_classes: Dot MathTex

        class ComplexPlaneExample(Scene):
            def construct(self):
                plane = ComplexPlane().add_coordinates()
                self.add(plane)
                d1 = Dot(plane.n2p(2 + 1j), color=YELLOW)
                d2 = Dot(plane.n2p(-3 - 2j), color=YELLOW)
                label1 = MathTex("2+i").next_to(d1, UR, 0.1)
                label2 = MathTex("-3-2i").next_to(d2, UR, 0.1)
                self.add(
                    d1,
                    label1,
                    d2,
                    label2,
                )

    """

    def __init__(self, **kwargs):
        super().__init__(
            **kwargs,
        )

    def number_to_point(self, number: float | complex) -> np.ndarray:
        """Accepts a float/complex number and returns the equivalent point on the plane.

        Parameters
        ----------
        number
            The number. Can be a float or a complex number.

        Returns
        -------
        np.ndarray
            The point on the plane.
        """

        number = complex(number)
        return self.coords_to_point(number.real, number.imag)

    def n2p(self, number: float | complex) -> np.ndarray:
        """Abbreviation for :meth:`number_to_point`."""
        return self.number_to_point(number)

    def point_to_number(self, point: Sequence[float]) -> complex:
        """Accepts a point and returns a complex number equivalent to that point on the plane.

        Parameters
        ----------
        point
            The point in manim's coordinate-system

        Returns
        -------
        complex
            A complex number consisting of real and imaginary components.
        """

        x, y = self.point_to_coords(point)
        return complex(x, y)

    def p2n(self, point: Sequence[float]) -> complex:
        """Abbreviation for :meth:`point_to_number`."""
        return self.point_to_number(point)

    def _get_default_coordinate_values(self) -> list[float | complex]:
        """Generate a list containing the numerical values of the plane's labels.

        Returns
        -------
        List[float | complex]
            A list of floats representing the x-axis and complex numbers representing the y-axis.
        """
        x_numbers = self.get_x_axis().get_tick_range()
        y_numbers = self.get_y_axis().get_tick_range()
        y_numbers = [complex(0, y) for y in y_numbers if y != 0]
        return [*x_numbers, *y_numbers]

    def get_coordinate_labels(
        self, *numbers: Iterable[float | complex], **kwargs
    ) -> VGroup:
        """Generates the :class:`~.DecimalNumber` mobjects for the coordinates of the plane.

        Parameters
        ----------
        numbers
            An iterable of floats/complex numbers. Floats are positioned along the x-axis, complex numbers along the y-axis.
        kwargs
            Additional arguments to be passed to :meth:`~.NumberLine.get_number_mobject`, i.e. :class:`~.DecimalNumber`.

        Returns
        -------
        :class:`~.VGroup`
            A :class:`~.VGroup` containing the positioned label mobjects.
        """

        # TODO: Make this work the same as coord_sys.add_coordinates()
        if len(numbers) == 0:
            numbers = self._get_default_coordinate_values()

        self.coordinate_labels = VGroup()
        for number in numbers:
            z = complex(number)
            if abs(z.imag) > abs(z.real):
                axis = self.get_y_axis()
                value = z.imag
                kwargs["unit"] = "i"
            else:
                axis = self.get_x_axis()
                value = z.real
            number_mob = axis.get_number_mobject(value, **kwargs)
            self.coordinate_labels.add(number_mob)
        return self.coordinate_labels

    def add_coordinates(self, *numbers: Iterable[float | complex], **kwargs):
        """Adds the labels produced from :meth:`~.NumberPlane.get_coordinate_labels` to the plane.

        Parameters
        ----------
        numbers
            An iterable of floats/complex numbers. Floats are positioned along the x-axis, complex numbers along the y-axis.
        kwargs
            Additional arguments to be passed to :meth:`~.NumberLine.get_number_mobject`, i.e. :class:`~.DecimalNumber`.
        """

        self.add(self.get_coordinate_labels(*numbers, **kwargs))
        return self
