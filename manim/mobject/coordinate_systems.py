"""Mobjects that represent coordinate systems."""


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
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from colour import Color

from manim.mobject.opengl_compatibility import ConvertToOpenGL

from .. import config
from ..constants import *
from ..mobject.functions import ParametricFunction
from ..mobject.geometry import (
    Arrow,
    Circle,
    DashedLine,
    Dot,
    Line,
    Rectangle,
    RegularPolygon,
)
from ..mobject.number_line import NumberLine
from ..mobject.svg.tex_mobject import MathTex
from ..mobject.types.vectorized_mobject import (
    Mobject,
    VDict,
    VectorizedPoint,
    VGroup,
    VMobject,
)
from ..utils.color import (
    BLACK,
    BLUE,
    BLUE_D,
    GREEN,
    LIGHT_GREY,
    WHITE,
    YELLOW,
    color_gradient,
    invert_color,
)
from ..utils.config_ops import merge_dicts_recursively, update_dict_recursively
from ..utils.simple_functions import binary_search
from ..utils.space_ops import angle_of_vector

# TODO: There should be much more code reuse between Axes, NumberPlane and GraphScene


class CoordinateSystem:
    """
    Abstract class for Axes and NumberPlane

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
                        "number_scale_value": 0.5,
                    },
                    tips=False,
                )

                # Labels for the x-axis and y-axis.
                y_label = grid.get_y_axis_label("y", edge=LEFT, direction=LEFT, buff=0.4)
                x_label = grid.get_x_axis_label("x")
                grid_labels = VGroup(x_label, y_label)

                graphs = VGroup()
                for n in np.arange(1, 20 + 0.5, 0.5):
                    graphs += grid.get_graph(lambda x: x ** n, color=WHITE)
                    graphs += grid.get_graph(
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
                    scale_factor=0.85,
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

    def c2p(self, *coords):
        """Abbreviation for coords_to_point"""
        return self.coords_to_point(*coords)

    def p2c(self, point):
        """Abbreviation for point_to_coords"""
        return self.point_to_coords(point)

    def get_axes(self):
        raise NotImplementedError()

    def get_axis(self, index):
        return self.get_axes()[index]

    def get_x_axis(self):
        return self.get_axis(0)

    def get_y_axis(self):
        return self.get_axis(1)

    def get_z_axis(self):
        return self.get_axis(2)

    def get_x_axis_label(self, label_tex, edge=UR, direction=UR, **kwargs):
        return self.get_axis_label(
            label_tex, self.get_x_axis(), edge, direction, **kwargs
        )

    def get_y_axis_label(
        self, label_tex, edge=UR, direction=UP * 0.5 + RIGHT, **kwargs
    ):
        return self.get_axis_label(
            label_tex, self.get_y_axis(), edge, direction, **kwargs
        )

    # move to a util_file, or Mobject()??
    @staticmethod
    def create_label_tex(label_tex) -> "Mobject":
        """Checks if the label is a ``float``, ``int`` or a ``str`` and creates a :class:`~.MathTex` label accordingly.

        Parameters
        ----------
        label_tex : The label to be compared against the above types.

        Returns
        -------
        :class:`~.Mobject`
            The label.
        """

        if (
            isinstance(label_tex, float)
            or isinstance(label_tex, int)
            or isinstance(label_tex, str)
        ):
            label_tex = MathTex(label_tex)
        return label_tex

    def get_axis_label(
        self,
        label: Union[float, str, "Mobject"],
        axis: "Mobject",
        edge: Sequence[float],
        direction: Sequence[float],
        buff: float = SMALL_BUFF,
    ) -> "Mobject":
        """Gets the label for an axis.

        Parameters
        ----------
        label
            The label. Can be any mobject or `int/float/str` to be used with :class:`~.MathTex`
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

        label = self.create_label_tex(label)
        label.next_to(axis.get_edge_center(edge), direction, buff=buff)
        label.shift_onto_screen(buff=MED_SMALL_BUFF)
        return label

    def get_axis_labels(
        self,
        x_label: Union[float, str, "Mobject"] = "x",
        y_label: Union[float, str, "Mobject"] = "y",
    ) -> "VGroup":
        """Defines labels for the x_axis and y_axis of the graph.

        Parameters
        ----------
        x_label
            The label for the x_axis
        y_label
            The label for the y_axis

        Returns
        -------
        :class:`~.VGroup`
            A :class:`~.Vgroup` of the labels for the x_axis and y_axis.

        See Also
        --------
        :class:`get_x_axis_label`
        :class:`get_y_axis_label`
        """

        self.axis_labels = VGroup(
            self.get_x_axis_label(x_label),
            self.get_y_axis_label(y_label),
        )
        return self.axis_labels

    def add_coordinates(
        self, *axes_numbers: Optional[Iterable[float]], **kwargs
    ) -> VGroup:
        """Adds labels to the axes.

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

        Returns
        -------
        VGroup
            A :class:`VGroup` of the number mobjects.
        """

        self.coordinate_labels = VGroup()
        # if nothing is passed to axes_numbers, produce axes with default labelling
        if not axes_numbers:
            axes_numbers = [None for _ in range(self.dimension)]

        for axis, values in zip(self.axes, axes_numbers):
            labels = axis.add_numbers(values, **kwargs)
            self.coordinate_labels.add(labels)

        return self.coordinate_labels

    def get_line_from_axis_to_point(
        self,
        index: int,
        point: Sequence[float],
        line_func: Line = DashedLine,
        color: Color = LIGHT_GREY,
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
        color
            The color of the line.
        stroke_width
            The stroke width of the line.

        Returns
        -------
        :class:`~.Line`
            The line from an axis to a point.

        See Also
        --------
        :class:`get_vertical_line`
        :class:`get_horizontal_line`
        """
        axis = self.get_axis(index)
        line = line_func(axis.get_projection(point), point)
        line.set_stroke(color, stroke_width)
        return line

    def get_vertical_line(self, point: Sequence[float], **kwargs) -> Line:
        """A vertical line from the x-axis to a given point in the scene.

        Parameters
        ----------
        point
            The point to which the vertical line will be drawn.

        kwargs
            Additional parameters to be passed to :class:`get_line_from_axis_to_point`

        Returns
        -------
        :class:`Line`
            A vertical line from the x-axis to the point.
        """

        return self.get_line_from_axis_to_point(0, point, **kwargs)

    def get_horizontal_line(self, point: Sequence[float], **kwargs) -> Line:
        """A horizontal line from the y-axis to a given point in the scene.

        Parameters
        ----------
        point
            The point to which the horizontal line will be drawn.

        kwargs
            Additional parameters to be passed to :class:`get_line_from_axis_to_point`

        Returns
        -------
        :class:`Line`
            A horizontal line from the y-axis to the point.
        """

        return self.get_line_from_axis_to_point(1, point, **kwargs)

    # graphing

    def get_graph(
        self,
        function: Callable[[float], float],
        x_range: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        """Generates a curve based on a function.

        Parameters
        ----------
        function
            The function used to construct the :class:`~.ParametricFunction`.

        x_range
            The range of the curve along the axes. ``x_range = [x_min, x_max]``.

        kwargs
            Additional parameters to be passed to :class:`~.ParametricFunction`.

        Returns
        -------
        :class:`~.ParametricFunction`
            The plotted curve.
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
            lambda t: self.coords_to_point(t, function(t)), t_range=t_range, **kwargs
        )
        graph.underlying_function = function
        return graph

    def get_parametric_curve(self, function, **kwargs):
        dim = self.dimension
        graph = ParametricFunction(
            lambda t: self.coords_to_point(*function(t)[:dim]), **kwargs
        )
        graph.underlying_function = function
        return graph

    def input_to_graph_point(self, x: float, graph: "ParametricFunction") -> np.ndarray:
        """Returns the coordinates of the point on the ``graph``
        corresponding to the input ``x`` value.

        Parameters
        ----------
        x
            The x-value for which the coordinates of corresponding point on the :attr:`graph` are to be found.

        graph
            The :class:`~.ParametricFunction` on which the x-value and y-value lie.

        Returns
        -------
        :class:`np.ndarray`
            The coordinates of the point on the :attr:`graph` corresponding to the :attr:`x` value.
        """

        if hasattr(graph, "underlying_function"):
            return graph.function(x)
        else:
            alpha = binary_search(
                function=lambda a: self.point_to_coords(graph.point_from_proportion(a))[
                    0
                ],
                target=x,
                lower_bound=self.x_range[0],
                upper_bound=self.x_range[1],
            )
            if alpha is not None:
                return graph.point_from_proportion(alpha)
            else:
                return None

    def i2gp(self, x, graph):
        """
        Alias for :meth:`input_to_graph_point`.
        """
        return self.input_to_graph_point(x, graph)

    def get_graph_label(
        self,
        graph: "ParametricFunction",
        label: Union[float, str, "Mobject"] = "f(x)",
        x_val: Optional[float] = None,
        direction: Sequence[float] = RIGHT,
        buff: float = MED_SMALL_BUFF,
        color: Optional[Color] = None,
        dot: bool = False,
        dot_config: Optional[dict] = None,
    ) -> Mobject:
        """Creates a properly positioned label for the passed graph,
        styled with parameters and an optional dot.

        Parameters
        ----------
        graph
            The curve of the function plotted.
        label
            The label for the function's curve. Written with :class:`MathTex` if not specified otherwise.
        x_val
            The x_value with which the label should be aligned.
        direction
            The cartesian position, relative to the curve that the label will be at --> ``LEFT``, ``RIGHT``
        buff
            The buffer space between the curve and the label.
        color
            The color of the label.
        dot
            Adds a dot at the given point on the graph.
        dot_config
            Additional parameters to be passed to :class:`~.Dot`.

        Returns
        -------
        :class:`Mobject`
            The positioned label and :class:`~.Dot`, if applicable.
        """

        if dot_config is None:
            dot_config = {}
        label = self.create_label_tex(label)
        color = color or graph.get_color()
        label.set_color(color)

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
            label.add(Dot(point=point, **dot_config))
        return label

    # calculus

    def get_riemann_rectangles(
        self,
        graph: "ParametricFunction",
        x_range: Optional[Sequence[float]] = None,
        dx: Optional[float] = 0.1,
        input_sample_type: str = "left",
        stroke_width: float = 1,
        stroke_color: Color = BLACK,
        fill_opacity: float = 1,
        color: Union[Iterable[Color], Color] = np.array((BLUE, GREEN)),
        show_signed_area: bool = True,
        bounded_graph: "ParametricFunction" = None,
        blend: bool = False,
        width_scale_factor: float = 1.001,
    ) -> VGroup:
        """This method returns the :class:`~.VGroup` of the Riemann Rectangles for
        a particular curve.

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
        """

        # setting up x_range, overwrite user's third input
        if x_range is None:
            x_range = self.x_range

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
                y_point = self.origin_shift(self.y_range)
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
                    )
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
        graph: "ParametricFunction",
        x_range: Optional[Sequence[float]] = None,
        color: Union[Color, Iterable[Color]] = [BLUE, GREEN],
        opacity: float = 0.3,
        dx_scaling: float = 1,
        bounded: "ParametricFunction" = None,
    ):
        """Returns a :class:`~.VGroup` of Riemann rectangles sufficiently small enough to visually
        approximate the area under the graph passed.

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

        bounded
            If a secondary :attr:`graph` is specified, encloses the area between the two curves.

        dx_scaling
            The factor by which the :attr:`dx` value is scaled.

        Returns
        -------
        :class:`~.VGroup`
            The :class:`~.VGroup` containing the Riemann Rectangles.
        """

        dx = self.x_range[2] / 500
        return self.get_riemann_rectangles(
            graph,
            x_range=x_range,
            dx=dx * dx_scaling,
            bounded_graph=bounded,
            blend=True,
            color=color,
            show_signed_area=False,
        ).set_opacity(opacity=opacity)

    def angle_of_tangent(
        self, x: float, graph: "ParametricFunction", dx: float = 1e-8
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
            The small change in `x` with which a small change in `y`
            will be compared in order to obtain the tangent.

        Returns
        -------
        :class:`float`
            The angle of the tangent with the x axis.
        """

        p0 = self.input_to_graph_point(x, graph)
        p1 = self.input_to_graph_point(x + dx, graph)
        return angle_of_vector(p1 - p0)

    def slope_of_tangent(
        self, x: float, graph: "ParametricFunction", **kwargs
    ) -> float:
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
        """

        return np.tan(self.angle_of_tangent(x, graph, **kwargs))

    def get_derivative_graph(
        self, graph: "ParametricFunction", color: Color = GREEN, **kwargs
    ) -> ParametricFunction:
        """Returns the curve of the derivative of the passed
        graph.

        Parameters
        ----------
        graph
            The graph for which the derivative will be found.

        color
            The color of the derivative curve.

        **kwargs
            Any valid keyword argument of :class:`~.ParametricFunction`

        Returns
        -------
        :class:`~.ParametricFunction`
            The curve of the derivative.
        """

        def deriv(x):
            return self.slope_of_tangent(x, graph)

        return self.get_graph(deriv, color=color, **kwargs)

    def get_secant_slope_group(
        self,
        x: float,
        graph: ParametricFunction,
        dx: Optional[float] = None,
        dx_line_color: Color = YELLOW,
        dy_line_color: Optional[Color] = None,
        dx_label: Optional[Union[float, str]] = None,
        dy_label: Optional[Union[float, str]] = None,
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
            The label for the `dx` line.

        dy_label
            The label for the `dy` line.

        include_secant_line
            Whether or not to include the secant line in the graph,
            or just have the df and dx lines and labels.

        secant_line_color
            The color of the secant line.

        secant_line_length
            The length of the secant line.

        Returns
        -------
        :class:`~.VGroup`
            A group containing the elements: `dx_line`, `df_line`, and
            if applicable also :attr:`dx_label`, :attr:`df_label`, `secant_line`.

        """
        group = VGroup()

        dx = dx or float(self.x_range[1] - self.x_range[0]) / 10
        dx_line_color = dx_line_color
        dy_line_color = dy_line_color or graph.get_color()

        p1 = self.input_to_graph_point(x, graph)
        p2 = self.input_to_graph_point(x + dx, graph)
        interim_point = p2[0] * RIGHT + p1[1] * UP

        group.dx_line = Line(p1, interim_point, color=dx_line_color)
        group.df_line = Line(interim_point, p2, color=dy_line_color)
        group.add(group.dx_line, group.df_line)

        labels = VGroup()
        if dx_label is not None:
            group.dx_label = self.create_label_tex(dx_label)
            labels.add(group.dx_label)
            group.add(group.dx_label)
        if dy_label is not None:
            group.df_label = self.create_label_tex(dy_label)
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
                group.dx_line, np.sign(dx) * DOWN, buff=group.dx_label.height / 2
            )
            group.dx_label.set_color(group.dx_line.get_color())

        if dy_label is not None:
            group.df_label.next_to(
                group.df_line, np.sign(dx) * RIGHT, buff=group.df_label.height / 2
            )
            group.df_label.set_color(group.df_line.get_color())

        if include_secant_line:
            secant_line_color = secant_line_color
            group.secant_line = Line(p1, p2, color=secant_line_color)
            group.secant_line.scale_in_place(
                secant_line_length / group.secant_line.get_length()
            )
            group.add(group.secant_line)
        return group

    def get_vertical_lines_to_graph(
        self,
        graph: ParametricFunction,
        x_range: Optional[Sequence[float]] = None,
        num_lines: int = 20,
        **kwargs,
    ) -> VGroup:
        """Obtains multiple lines from the x-axis to the curve.

        Parameters
        ----------
        graph
            The graph on which the line should extend to.

        x_range
            A list containing the lower and and upper bounds of the lines -> ``x_range = [x_min, x_max]``.

        num_lines
            The number of evenly spaced lines.

        Returns
        -------
        :class:`~.VGroup`
            The :class:`~.VGroup` of the evenly spaced lines.
        """

        x_range = x_range if x_range is not None else self.x_range

        return VGroup(
            *[
                self.get_vertical_line(self.i2gp(x, graph), **kwargs)
                for x in np.linspace(x_range[0], x_range[1], num_lines)
            ]
        )

    def get_T_label(
        self,
        x_val: float,
        graph: "ParametricFunction",
        label: Optional[Union[float, str, "Mobject"]] = None,
        label_color: Color = WHITE,
        triangle_size: float = MED_SMALL_BUFF,
        triangle_color: Color = WHITE,
        line_func: "Line" = Line,
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

        Examples
        -------
        .. manim:: T_labelExample
            :save_last_frame:

            class T_labelExample(Scene):
                def construct(self):
                    # defines the axes and linear function
                    axes = Axes(x_range=[-1, 10], y_range=[-1, 10], x_length=9, y_length=6)
                    func = axes.get_graph(lambda x: x, color=BLUE)
                    # creates the T_label
                    t_label = axes.get_T_label(x_val=4, graph=func, label=Tex("x-value"))
                    self.add(axes, func, t_label)

        Returns
        -------
        :class:`~.VGroup`
            A :class:`~.VGroup` of the label, triangle and vertical line mobjects.
        """

        T_label_group = VGroup()
        triangle = RegularPolygon(n=3, start_angle=np.pi / 2, stroke_width=0).set_fill(
            color=triangle_color, opacity=1
        )
        triangle.height = triangle_size
        triangle.move_to(self.coords_to_point(x_val, 0), UP)
        if label is not None:
            t_label = self.create_label_tex(label).set_color(label_color)
            t_label.next_to(triangle, DOWN)
            T_label_group.add(t_label)

        v_line = self.get_vertical_line(
            self.i2gp(x_val, graph), color=line_color, line_func=line_func
        )

        T_label_group.add(triangle, v_line)

        return T_label_group


class Axes(VGroup, CoordinateSystem, metaclass=ConvertToOpenGL):
    """Creates a set of axes.

    Parameters
    ----------
    x_range
        The :code:`[x_min, x_max, x_step]` values of the x-axis.
    y_range
        The :code:`[y_min, y_max, y_step]` values of the y-axis.
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
    kwargs : Any
        Additional arguments to be passed to :class:`CoordinateSystem` and :class:`~.VGroup`.
    """

    def __init__(
        self,
        x_range: Optional[Sequence[float]] = None,
        y_range: Optional[Sequence[float]] = None,
        x_length: Optional[float] = round(config.frame_width) - 2,
        y_length: Optional[float] = round(config.frame_height) - 2,
        axis_config: Optional[dict] = None,
        x_axis_config: Optional[dict] = None,
        y_axis_config: Optional[dict] = None,
        tips: bool = True,
        **kwargs,
    ):
        VGroup.__init__(self, **kwargs)
        CoordinateSystem.__init__(self, x_range, y_range, x_length, y_length)

        self.axis_config = {
            "include_tip": tips,
            "numbers_to_exclude": [0],
            "exclude_origin_tick": True,
        }
        self.x_axis_config = {}
        self.y_axis_config = {"rotation": 90 * DEGREES, "label_direction": LEFT}

        self.update_default_configs(
            (self.axis_config, self.x_axis_config, self.y_axis_config),
            (axis_config, x_axis_config, y_axis_config),
        )

        self.x_axis_config = merge_dicts_recursively(
            self.axis_config, self.x_axis_config
        )
        self.y_axis_config = merge_dicts_recursively(
            self.axis_config, self.y_axis_config
        )

        self.x_axis = self.create_axis(self.x_range, self.x_axis_config, self.x_length)
        self.y_axis = self.create_axis(self.y_range, self.y_axis_config, self.y_length)

        # Add as a separate group in case various other
        # mobjects are added to self, as for example in
        # NumberPlane below
        self.axes = VGroup(self.x_axis, self.y_axis)
        self.add(*self.axes)

        # finds the middle-point on each axis
        lines_center_point = [((axis.x_max + axis.x_min) / 2) for axis in self.axes]

        self.shift(-self.coords_to_point(*lines_center_point))

    @staticmethod
    def update_default_configs(default_configs, passed_configs):
        for default_config, passed_config in zip(default_configs, passed_configs):
            if passed_config is not None:
                update_dict_recursively(default_config, passed_config)

    def create_axis(
        self,
        range_terms: Sequence[float],
        axis_config: dict,
        length: float,
    ) -> NumberLine:
        """Creates an axis and dynamically adjusts its position depending on where 0 is located on the line.

        Parameters
        ----------
        range_terms
            The range of the the axis : `(x_min, x_max, x_step)`.
        axis_config
            Additional parameters that are passed to :class:`NumberLine`.
        length
            The length of the axis.

        Returns
        -------
        :class:`NumberLine`
            Returns a number line with the provided x and y axis range.
        """
        axis_config["length"] = length
        axis = NumberLine(range_terms, **axis_config)

        # without the call to origin_shift, graph does not exist when min > 0 or max < 0
        # shifts the axis so that 0 is centered
        axis.shift(-axis.number_to_point(self.origin_shift(range_terms)))
        return axis

    def coords_to_point(self, *coords: Sequence[float]) -> np.ndarray:
        """Transforms the vector formed from ``coords`` formed by the :class:`Axes`
        into the corresponding vector with respect to the default basis.

        Returns
        -------
        np.ndarray
            A point that results from a change of basis from the coordinate system
            defined by the :class:`Axes` to that of ``manim``'s default coordinate system
        """
        origin = self.x_axis.number_to_point(self.origin_shift(self.x_range))
        result = np.array(origin)
        for axis, coord in zip(self.get_axes(), coords):
            result += axis.number_to_point(coord) - origin
        return result

    def point_to_coords(self, point: float) -> Tuple:
        """Transforms the coordinates of the point which are with respect to ``manim``'s default
        basis into the coordinates of that point with respect to the basis defined by :class:`Axes`.

        Parameters
        ----------
        point
            The point whose coordinates will be found.

        Returns
        -------
        Tuple
            Coordinates of the point with respect to :class:`Axes`'s basis
        """
        return tuple([axis.point_to_number(point) for axis in self.get_axes()])

    def get_axes(self) -> VGroup:
        """Gets the axes.

        Returns
        -------
        :class:`~.VGroup`
            A pair of axes.
        """
        return self.axes

    def get_line_graph(
        self,
        x_values: Iterable[float],
        y_values: Iterable[float],
        z_values: Optional[Iterable[float]] = None,
        line_color: Color = YELLOW,
        add_vertex_dots: bool = True,
        vertex_dot_radius: float = DEFAULT_DOT_RADIUS,
        vertex_dot_style: Optional[dict] = None,
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
                    line_graph = plane.get_line_graph(
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
        graph = VMobject(color=line_color, **kwargs)
        vertices = [
            self.coords_to_point(x, y, z)
            for x, y, z in zip(x_values, y_values, z_values)
        ]
        graph.set_points_as_corners(vertices)
        graph.z_index = -1
        line_graph["line_graph"] = graph

        if add_vertex_dots:
            vertex_dot_style = vertex_dot_style or {}
            vertex_dots = VGroup(
                *[
                    Dot(point=vertex, radius=vertex_dot_radius, **vertex_dot_style)
                    for vertex in vertices
                ]
            )
            line_graph["vertex_dots"] = vertex_dots

        return line_graph

    @staticmethod
    def origin_shift(axis_range: Sequence[float]) -> float:
        """Determines how to shift graph mobjects to compensate when 0 is not on the axis.

        Parameters
        ----------
        axis_range
            The range of the axis : ``(x_min, x_max, x_step)``.
        """
        if axis_range[0] > 0:
            return axis_range[0]
        if axis_range[1] < 0:
            return axis_range[1]
        else:
            return 0


class ThreeDAxes(Axes):
    """A 3-dimensional set of axes.

    Parameters
    ----------
    x_range
        The :code:`[x_min, x_max, x_step]` values of the x-axis.
    y_range
        The :code:`[y_min, y_max, y_step]` values of the y-axis.
    z_range
        The :code:`[z_min, z_max, z_step]` values of the z-axis.
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
    kwargs : Any
        Additional arguments to be passed to :class:`Axes`.
    """

    def __init__(
        self,
        x_range: Optional[Sequence[float]] = (-6, 6, 1),
        y_range: Optional[Sequence[float]] = (-5, 5, 1),
        z_range: Optional[Sequence[float]] = (-4, 4, 1),
        x_length: Optional[float] = config.frame_height + 2.5,
        y_length: Optional[float] = config.frame_height + 2.5,
        z_length: Optional[float] = config.frame_height - 1.5,
        z_axis_config: Optional[dict] = None,
        z_normal: Sequence[float] = DOWN,
        num_axis_pieces: int = 20,
        light_source: Sequence[float] = 9 * DOWN + 7 * LEFT + 10 * OUT,
        # opengl stuff (?)
        depth=None,
        gloss=0.5,
        **kwargs,
    ):

        Axes.__init__(
            self,
            x_range=x_range,
            x_length=x_length,
            y_range=y_range,
            y_length=y_length,
            **kwargs,
        )

        self.z_range = z_range
        self.z_length = z_length

        self.z_axis_config = {}
        self.update_default_configs((self.z_axis_config,), (z_axis_config,))
        self.z_axis_config = merge_dicts_recursively(
            self.axis_config, self.z_axis_config
        )

        self.z_normal = z_normal
        self.num_axis_pieces = num_axis_pieces

        self.light_source = light_source

        self.dimension = 3

        z_axis = self.create_axis(self.z_range, self.z_axis_config, self.z_length)

        z_axis.rotate_about_zero(-PI / 2, UP)
        z_axis.rotate_about_zero(angle_of_vector(self.z_normal))
        z_axis.shift(self.x_axis.number_to_point(self.origin_shift(x_range)))

        self.axes.add(z_axis)
        self.add(z_axis)
        self.z_axis = z_axis

        if not config.renderer == "opengl":
            self.add_3d_pieces()
            self.set_axis_shading()

    def add_3d_pieces(self):
        for axis in self.axes:
            axis.pieces = VGroup(*axis.get_pieces(self.num_axis_pieces))
            axis.add(axis.pieces)
            axis.set_stroke(width=0, family=False)
            axis.set_shade_in_3d(True)

    def set_axis_shading(self):
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


class NumberPlane(Axes):
    """Creates a cartesian plane with background lines.

    Parameters
    ----------
    x_range
        The :code:`[x_min, x_max, x_step]` values of the plane in the horizontal direction.
    y_range
        The :code:`[y_min, y_max, y_step]` values of the plane in the vertical direction.
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
    kwargs : Any
        Additional arguments to be passed to :class:`Axes`.

    .. note:: If :attr:`x_length` or :attr:`y_length` are not defined, the plane automatically adjusts its lengths based
        on the :attr:`x_range` and :attr:`y_range` values to set the unit_size to 1.

    Examples
    --------

    .. manim:: NumberPlaneExample
        :save_last_frame:

        class NumberPlaneExample(Scene):
            def construct(self):
                number_plane = NumberPlane(
                    x_range=[-10, 10, 1],
                    y_range=[-10, 10, 1],
                    background_line_style={
                        "stroke_color": TEAL,
                        "stroke_width": 4,
                        "stroke_opacity": 0.6
                    }
                )
                self.add(number_plane)
    """

    def __init__(
        self,
        x_range: Optional[Sequence[float]] = (
            -config["frame_x_radius"],
            config["frame_x_radius"],
            1,
        ),
        y_range: Optional[Sequence[float]] = (
            -config["frame_y_radius"],
            config["frame_y_radius"],
            1,
        ),
        x_length: Optional[float] = None,
        y_length: Optional[float] = None,
        background_line_style: Optional[dict] = None,
        faded_line_style: Optional[dict] = None,
        faded_line_ratio: Optional[float] = 1,
        make_smooth_after_applying_functions=True,
        **kwargs,
    ):

        # configs
        self.axis_config = {
            "stroke_color": WHITE,
            "stroke_width": 2,
            "include_ticks": False,
            "include_tip": False,
            "line_to_number_buff": SMALL_BUFF,
            "label_direction": DR,
            "number_scale_value": 0.5,
        }
        self.y_axis_config = {"label_direction": DR}
        self.background_line_style = {
            "stroke_color": BLUE_D,
            "stroke_width": 2,
            "stroke_opacity": 1,
        }

        self.update_default_configs(
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

        # dynamically adjusts x_length and y_length so that the unit_size is one by default
        if x_length is None:
            x_length = self.x_range[1] - self.x_range[0]
        if y_length is None:
            y_length = self.y_range[1] - self.y_range[0]

        self.init_background_lines()

    def init_background_lines(self):
        """Will init all the lines of NumberPlanes (faded or not)"""
        if self.faded_line_style is None:
            style = dict(self.background_line_style)
            # For anything numerical, like stroke_width
            # and stroke_opacity, chop it in half
            for key in style:
                if isinstance(style[key], numbers.Number):
                    style[key] *= 0.5
            self.faded_line_style = style

        self.background_lines, self.faded_lines = self.get_lines()
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

    def get_lines(self) -> Tuple[VGroup, VGroup]:
        """Generate all the lines, faded and not faded. Two sets of lines are generated: one parallel to the X-axis, and parallel to the Y-axis.

        Returns
        -------
        Tuple[:class:`~.VGroup`, :class:`~.VGroup`]
            The first (i.e the non faded lines) and second (i.e the faded lines) sets of lines, respectively.
        """
        x_axis = self.get_x_axis()
        y_axis = self.get_y_axis()

        x_lines1, x_lines2 = self.get_lines_parallel_to_axis(
            x_axis,
            y_axis,
            self.x_axis.x_step,
            self.faded_line_ratio,
        )
        y_lines1, y_lines2 = self.get_lines_parallel_to_axis(
            y_axis,
            x_axis,
            self.y_axis.x_step,
            self.faded_line_ratio,
        )
        lines1 = VGroup(*x_lines1, *y_lines1)
        lines2 = VGroup(*x_lines2, *y_lines2)
        return lines1, lines2

    def get_lines_parallel_to_axis(
        self,
        axis_parallel_to: Line,
        axis_perpendicular_to: Line,
        freq: float,
        ratio_faded_lines: float,
    ) -> Tuple[VGroup, VGroup]:
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
            The first (i.e the non-faded lines parallel to `axis_parallel_to`) and second (i.e the faded lines parallel to `axis_parallel_to`) sets of lines, respectively.
        """

        line = Line(axis_parallel_to.get_start(), axis_parallel_to.get_end())
        if ratio_faded_lines == 0:  # don't show faded lines
            ratio_faded_lines = 1  # i.e. set ratio to 1
        step = (1 / ratio_faded_lines) * freq
        lines1 = VGroup()
        lines2 = VGroup()
        unit_vector_axis_perp_to = axis_perpendicular_to.get_unit_vector()
        ranges = (
            np.arange(0, axis_perpendicular_to.x_max, step),
            np.arange(0, axis_perpendicular_to.x_min, -step),
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

    def get_center_point(self) -> np.ndarray:
        """Gets the origin of :class:`NumberPlane`.

        Returns
        -------
        np.ndarray
            The center point.
        """
        return self.coords_to_point(0, 0)

    def get_x_unit_size(self):
        return self.get_x_axis().get_unit_size()

    def get_y_unit_size(self):
        return self.get_x_axis().get_unit_size()

    def get_axes(self) -> VGroup:
        # Method Already defined at Axes.get_axes so we could remove this a later PR.
        """Gets the pair of axes.

        Returns
        -------
        :class:`~.VGroup`
            Axes
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
        If the ``azimuth_units`` choice has fractional labels, choose whether to combine the constant in a compact form :math:`\tfrac{xu}{y}` as opposed to :math:`\tfrac{x}{y}u`, where :math:`u` is the constant.

    azimuth_offset
        The angle offset of the azimuth, expressed in radians.

    azimuth_direction
        The direction of the azimuth.

        - ``"CW"``: Clockwise.
        - ``"CCW"``: Anti-clockwise.

    azimuth_label_buff
        The buffer for the azimuth labels.

    azimuth_label_scale
        The scale of the azimuth labels.

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
                    azimuth_label_scale=0.7,
                    radius_config={"number_scale_value": 0.7},
                ).add_coordinates()
                self.add(polarplane_pi)
    """

    def __init__(
        self,
        radius_max: float = config["frame_y_radius"],
        size: Optional[float] = None,
        radius_step: float = 1,
        azimuth_step: Optional[float] = None,
        azimuth_units: Optional[str] = "PI radians",
        azimuth_compact_fraction: bool = True,
        azimuth_offset: float = 0,
        azimuth_direction: str = "CCW",
        azimuth_label_buff: float = SMALL_BUFF,
        azimuth_label_scale: float = 0.5,
        radius_config: Optional[dict] = None,
        background_line_style: Optional[dict] = None,
        faded_line_style: Optional[dict] = None,
        faded_line_ratio: int = 1,
        make_smooth_after_applying_functions: bool = True,
        **kwargs,
    ):

        # error catching
        if azimuth_units in ["PI radians", "TAU radians", "degrees", "gradians", None]:
            self.azimuth_units = azimuth_units
        else:
            raise ValueError(
                "Invalid azimuth units. Expected one of: PI radians, TAU radians, degrees, gradians or None."
            )

        if azimuth_direction in ["CW", "CCW"]:
            self.azimuth_direction = azimuth_direction
        else:
            raise ValueError("Invalid azimuth units. Expected one of: CW, CCW.")

        # configs
        self.radius_config = {
            "stroke_color": WHITE,
            "stroke_width": 2,
            "include_ticks": False,
            "include_tip": False,
            "line_to_number_buff": SMALL_BUFF,
            "label_direction": DL,
            "number_scale_value": 0.5,
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

        self.update_default_configs(
            (self.radius_config, self.background_line_style),
            (radius_config, background_line_style),
        )

        # Defaults to a faded version of line_config
        self.faded_line_style = faded_line_style
        self.faded_line_ratio = faded_line_ratio
        self.make_smooth_after_applying_functions = make_smooth_after_applying_functions
        self.azimuth_offset = azimuth_offset
        self.azimuth_label_buff = azimuth_label_buff
        self.azimuth_label_scale = azimuth_label_scale
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

        # dynamically adjusts size so that the unit_size is one by default
        if size is None:
            size = 0

        self.init_background_lines()

    def init_background_lines(self):
        """Will init all the lines of NumberPlanes (faded or not)"""
        if self.faded_line_style is None:
            style = dict(self.background_line_style)
            # For anything numerical, like stroke_width
            # and stroke_opacity, chop it in half
            for key in style:
                if isinstance(style[key], numbers.Number):
                    style[key] *= 0.5
            self.faded_line_style = style

        self.background_lines, self.faded_lines = self.get_lines()
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

    def get_lines(self) -> Tuple[VGroup, VGroup]:
        """Generate all the lines and circles, faded and not faded.

        Returns
        -------
        Tuple[:class:`~.VGroup`, :class:`~.VGroup`]
            The first (i.e the non faded lines and circles) and second (i.e the faded lines and circles) sets of lines and circles, respectively.
        """
        center = self.get_center_point()
        ratio_faded_lines = self.faded_line_ratio
        offset = self.azimuth_offset

        if ratio_faded_lines == 0:  # don't show faded lines
            ratio_faded_lines = 1  # i.e. set ratio to 1
        rstep = (1 / ratio_faded_lines) * self.x_axis.x_step
        astep = (1 / ratio_faded_lines) * (TAU * (1 / self.azimuth_step))
        rlines1 = VGroup()
        rlines2 = VGroup()
        alines1 = VGroup()
        alines2 = VGroup()

        rinput = np.arange(0, self.x_axis.x_max + rstep, rstep)
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

    def get_center_point(self):
        return self.coords_to_point(0, 0)

    def get_x_unit_size(self):
        return self.get_x_axis().get_unit_size()

    def get_y_unit_size(self):
        return self.get_x_axis().get_unit_size()

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

    def pr2pt(self, radius: float, azimuth: float) -> np.ndarray:
        """Abbreviation for :meth:`polar_to_point`"""
        return self.polar_to_point(radius, azimuth)

    def point_to_polar(self, point: np.ndarray) -> Tuple[float, float]:
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
        return np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)

    def pt2pr(self, point: np.ndarray) -> Tuple[float, float]:
        """Abbreviation for :meth:`point_to_polar`"""
        return self.point_to_polar(point)

    def get_coordinate_labels(
        self,
        r_values: Optional[Iterable[float]] = None,
        a_values: Optional[Iterable[float]] = None,
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
                    ]
                ),
            }
            for i in a_values
        ]
        if self.azimuth_units == "PI radians" or self.azimuth_units == "TAU radians":
            a_tex = [
                self.get_radian_label(i["label"])
                .scale(self.azimuth_label_scale)
                .next_to(
                    i["point"],
                    direction=i["point"],
                    aligned_edge=i["point"],
                    buff=self.azimuth_label_buff,
                )
                for i in a_points
            ]
        elif self.azimuth_units == "degrees":
            a_tex = [
                MathTex(f'{360 * i["label"]:g}' + r"^{\circ}")
                .scale(self.azimuth_label_scale)
                .next_to(
                    i["point"],
                    direction=i["point"],
                    aligned_edge=i["point"],
                    buff=self.azimuth_label_buff,
                )
                for i in a_points
            ]
        elif self.azimuth_units == "gradians":
            a_tex = [
                MathTex(f'{400 * i["label"]:g}' + r"^{g}")
                .scale(self.azimuth_label_scale)
                .next_to(
                    i["point"],
                    direction=i["point"],
                    aligned_edge=i["point"],
                    buff=self.azimuth_label_buff,
                )
                for i in a_points
            ]
        elif self.azimuth_units is None:
            a_tex = [
                MathTex(f'{i["label"]:g}')
                .scale(self.azimuth_label_scale)
                .next_to(
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
        r_values: Optional[Iterable[float]] = None,
        a_values: Optional[Iterable[float]] = None,
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

    def get_radian_label(self, number, stacked=True):
        constant_label = {"PI radians": r"\pi", "TAU radians": r"\tau"}[
            self.azimuth_units
        ]
        division = number * {"PI radians": 2, "TAU radians": 1}[self.azimuth_units]
        frac = fr.Fraction(division).limit_denominator(max_denominator=100)
        if frac.numerator == 0 & frac.denominator == 0:
            return MathTex(r"0")
        elif frac.numerator == 1 and frac.denominator == 1:
            return MathTex(constant_label)
        elif frac.numerator == 1:
            if self.azimuth_compact_fraction:
                return MathTex(
                    r"\tfrac{" + constant_label + r"}{" + str(frac.denominator) + "}"
                )
            else:
                return MathTex(
                    r"\tfrac{1}{" + str(frac.denominator) + "}" + constant_label
                )
        elif frac.denominator == 1:
            return MathTex(str(frac.numerator) + constant_label)
        else:
            if self.azimuth_compact_fraction:
                return MathTex(
                    r"\tfrac{"
                    + str(frac.numerator)
                    + constant_label
                    + r"}{"
                    + str(frac.denominator)
                    + r"}"
                )
            else:
                return MathTex(
                    r"\tfrac{"
                    + str(frac.numerator)
                    + r"}{"
                    + str(frac.denominator)
                    + r"}"
                    + constant_label
                )


class ComplexPlane(NumberPlane):
    """
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

    def __init__(self, color=BLUE, **kwargs):
        super().__init__(
            color=color,
            **kwargs,
        )

    def number_to_point(self, number):
        number = complex(number)
        return self.coords_to_point(number.real, number.imag)

    def n2p(self, number):
        return self.number_to_point(number)

    def point_to_number(self, point):
        x, y = self.point_to_coords(point)
        return complex(x, y)

    def p2n(self, point):
        return self.point_to_number(point)

    def get_default_coordinate_values(self):
        x_numbers = self.get_x_axis().get_tick_range()
        y_numbers = self.get_y_axis().get_tick_range()
        y_numbers = [complex(0, y) for y in y_numbers if y != 0]
        return [*x_numbers, *y_numbers]

    def get_coordinate_labels(self, *numbers, **kwargs):
        if len(numbers) == 0:
            numbers = self.get_default_coordinate_values()

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

    def add_coordinates(self, *numbers):
        self.add(self.get_coordinate_labels(*numbers))
        return self
