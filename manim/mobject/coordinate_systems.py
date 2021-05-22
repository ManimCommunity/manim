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
import math
import numbers
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .. import config
from ..constants import *
from ..mobject.functions import ParametricFunction
from ..mobject.geometry import Arrow, Circle, DashedLine, Dot, Line
from ..mobject.number_line import NumberLine
from ..mobject.svg.tex_mobject import MathTex
from ..mobject.types.vectorized_mobject import VDict, VGroup, VMobject
from ..utils.color import BLUE, BLUE_D, LIGHT_GREY, WHITE, YELLOW, Colors
from ..utils.config_ops import merge_dicts_recursively, update_dict_recursively
from ..utils.simple_functions import binary_search
from ..utils.space_ops import angle_of_vector

# TODO: There should be much more code reuse between Axes, NumberPlane and GraphScene


class CoordinateSystem:
    """
    Abstract class for Axes and NumberPlane
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

        if x_range is None:
            x_range = [
                round(-config["frame_x_radius"]),
                round(config["frame_x_radius"]),
                1.0,
            ]
        if y_range is None:
            y_range = [
                round(-config["frame_y_radius"]),
                round(config["frame_y_radius"]),
                1.0,
            ]

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

    def get_x_axis_label(
        self, label_tex, edge=RIGHT, direction=UP * 4 + RIGHT, **kwargs
    ):
        return self.get_axis_label(
            label_tex, self.get_x_axis(), edge, direction, **kwargs
        )

    def get_y_axis_label(self, label_tex, edge=UP, direction=UP + RIGHT * 2, **kwargs):
        return self.get_axis_label(
            label_tex, self.get_y_axis(), edge, direction, **kwargs
        )

    def get_axis_label(self, label_tex, axis, edge, direction, buff=SMALL_BUFF):
        label = MathTex(label_tex)
        label.next_to(axis.get_edge_center(edge), direction, buff=buff)
        label.shift_onto_screen(buff=MED_SMALL_BUFF)
        return label

    def get_axis_labels(self, x_label_tex="x", y_label_tex="y"):
        self.axis_labels = VGroup(
            self.get_x_axis_label(x_label_tex),
            self.get_y_axis_label(y_label_tex),
        )
        return self.axis_labels

    def get_line_from_axis_to_point(
        self, index, point, line_func=DashedLine, color=LIGHT_GREY, stroke_width=2
    ):
        axis = self.get_axis(index)
        line = line_func(axis.get_projection(point), point)
        line.set_stroke(color, stroke_width)
        return line

    def get_vertical_line(self, point, **kwargs):
        return self.get_line_from_axis_to_point(0, point, **kwargs)

    def get_horizontal_line(self, point, **kwargs):
        return self.get_line_from_axis_to_point(1, point, **kwargs)

    # graphing

    def get_graph(self, function, **kwargs):
        t_range = self.x_range

        if len(t_range) == 3:
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

    def input_to_graph_point(self, x, graph):
        if hasattr(graph, "underlying_function"):
            return self.coords_to_point(x, graph.underlying_function(x))
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


class Axes(VGroup, CoordinateSystem):
    """Creates a set of axes.

    Parameters
    ----------
    x_range :
        The :code:`[x_min, x_max, x_step]` values of the x-axis.
    y_range :
        The :code:`[y_min, y_max, y_step]` values of the y-axis.
    x_length : Optional[:class:`float`]
        The length of the x-axis.
    y_length : Optional[:class:`float`]
        The length of the y-axis.
    axis_config : Optional[:class:`dict`]
        Arguments to be passed to :class:`~.NumberLine` that influences both axes.
    x_axis_config : Optional[:class:`dict`]
        Arguments to be passed to :class:`~.NumberLine` that influence the x-axis.
    y_axis_config : Optional[:class:`dict`]
        Arguments to be passed to :class:`~.NumberLine` that influence the y-axis.
    kwargs : Any
        Additional arguments to be passed to :class:`CoordinateSystem` and :class:`~.VGroup`.
    """

    def __init__(
        self,
        x_range: Optional[Sequence[float]] = None,
        y_range: Optional[Sequence[float]] = None,
        x_length=round(config.frame_width) - 2,
        y_length=round(config.frame_height) - 2,
        axis_config=None,
        x_axis_config=None,
        y_axis_config=None,
        **kwargs,
    ):
        VGroup.__init__(self, **kwargs)
        CoordinateSystem.__init__(self, x_range, y_range, x_length, y_length)

        self.axis_config = {"include_tip": True, "numbers_to_exclude": [0]}
        self.x_axis_config = {}
        self.y_axis_config = {"rotation": 90 * DEGREES, "label_direction": LEFT}

        self.update_default_configs(
            (self.axis_config, self.x_axis_config, self.y_axis_config),
            (axis_config, x_axis_config, y_axis_config),
        )
        self.x_axis = self.create_axis(self.x_range, self.x_axis_config, self.x_length)
        self.y_axis = self.create_axis(self.y_range, self.y_axis_config, self.y_length)

        # Add as a separate group in case various other
        # mobjects are added to self, as for example in
        # NumberPlane below
        self.axes = VGroup(self.x_axis, self.y_axis)
        self.add(*self.axes)
        self.center()

    @staticmethod
    def update_default_configs(default_configs, passed_configs):
        for default_config, passed_config in zip(default_configs, passed_configs):
            if passed_config is not None:
                update_dict_recursively(default_config, passed_config)

    def create_axis(self, range_terms, axis_config, length):
        """Creates an axis and dynamically adjusts its position depending on where 0 is located on the line.

        Parameters
        ----------
        range_terms : Union[:class:`list`, :class:`numpy.ndarray`]
            The range of the the axis : `(x_min, x_max, x_step)`.
        axis_config : :class:`dict`
            Additional parameters that are passed to :class:`NumberLine`.
        length : :class:`float`
            The length of the axis.
        """
        new_config = merge_dicts_recursively(self.axis_config, axis_config)
        new_config["length"] = length
        axis = NumberLine(range_terms, **new_config)

        # without the call to origin_shift, graph does not exist when min > 0 or max < 0
        # shifts the axis so that 0 is centered
        axis.shift(-axis.number_to_point(self.origin_shift(range_terms)))
        return axis

    def coords_to_point(self, *coords):
        origin = self.x_axis.number_to_point(self.origin_shift(self.x_range))
        result = np.array(origin)
        for axis, coord in zip(self.get_axes(), coords):
            result += axis.number_to_point(coord) - origin
        return result

    def point_to_coords(self, point):
        return tuple([axis.point_to_number(point) for axis in self.get_axes()])

    def get_axes(self):
        return self.axes

    def get_coordinate_labels(self, x_values=None, y_values=None, **kwargs):
        axes = self.get_axes()
        self.coordinate_labels = VGroup()
        for axis, values in zip(axes, [x_values, y_values]):
            labels = axis.add_numbers(values, **kwargs)
            self.coordinate_labels.add(labels)
        return self.coordinate_labels

    def add_coordinates(self, x_values=None, y_values=None):
        self.add(self.get_coordinate_labels(x_values, y_values))
        return self

    def get_line_graph(
        self,
        x_values: Iterable[float],
        y_values: Iterable[float],
        z_values: Optional[Iterable[float]] = None,
        line_color: Colors = YELLOW,
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
    def origin_shift(axis_range: List[float]) -> float:
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
    x_range :
        The :code:`[x_min, x_max, x_step]` values of the x-axis.
    y_range :
        The :code:`[y_min, y_max, y_step]` values of the y-axis.
    z_range :
        The :code:`[z_min, z_max, z_step]` values of the z-axis.
    x_length : Optional[:class:`float`]
        The length of the x-axis.
    y_length : Optional[:class:`float`]
        The length of the y-axis.
    z_length : Optional[:class:`float`]
        The length of the z-axis.
    z_axis_config : Optional[:class:`dict`]
        Arguments to be passed to :class:`~.NumberLine` that influence the z-axis.
    z_normal : Union[:class:`list`, :class:`numpy.ndarray`]
        The direction of the normal.
    num_axis_pieces : :class:`int`
        The number of pieces used to construct the axes.
    light_source : Union[:class:`list`, :class:`numpy.ndarray`]
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
        x_length=config.frame_height + 2.5,
        y_length=config.frame_height + 2.5,
        z_length=config.frame_height - 1.5,
        z_axis_config=None,
        z_normal=DOWN,
        num_axis_pieces=20,
        light_source=9 * DOWN + 7 * LEFT + 10 * OUT,
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
    x_range :
        The :code:`[x_min, x_max, x_step]` values of the plane in the horizontal direction.
    y_range :
        The :code:`[y_min, y_max, y_step]` values of the plane in the vertical direction.
    x_length : Optional[:class:`float`]
        The width of the plane.
    y_length : Optional[:class:`float`]
        The height of the plane.
    axis_config : Optional[:class:`dict`]
        Arguments to be passed to :class:`~.NumberLine` that influences both axes.
    y_axis_config : Optional[:class:`dict`]
        Arguments to be passed to :class:`~.NumberLine` that influence the y-axis.
    background_line_style : Optional[:class:`dict`]
        Arguments that influence the construction of the background lines of the plane.
    faded_line_style : Optional[:class:`dict`]
        Similar to :attr:`background_line_style`, affects the construction of the scene's background lines.
    faded_line_ratio : Optional[:class:`int`]
        Determines the number of boxes within the background lines: :code:`2` = 4 boxes, :code:`3` = 9 boxes.
    make_smooth_after_applying_functions
        Currently non-functional.
    kwargs : Any
        Additional arguments to be passed to :class:`Axes`.

    .. note:: If :attr:`x_length` or :attr:`y_length` are not defined, the plane automatically adjusts its lengths based
        on the :attr:`x_range` and :attr:`y_range` values to set the unit_size to 1.
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
        x_length=None,
        y_length=None,
        axis_config=None,
        y_axis_config=None,
        background_line_style=None,
        faded_line_style=None,
        faded_line_ratio=1,
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
            (axis_config, y_axis_config, background_line_style),
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

    def get_lines(self):
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
        self, axis_parallel_to, axis_perpendicular_to, freq, ratio_faded_lines
    ):
        """Generate a set of lines parallel to an axis.

        Parameters
        ----------
        axis_parallel_to : :class:`~.Line`
            The axis with which the lines will be parallel.

        axis_perpendicular_to : :class:`~.Line`
            The axis with which the lines will be perpendicular.

        ratio_faded_lines : :class:`float`
            The ratio between the space between faded lines and the space between non-faded lines.

        freq : :class:`float`
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

    def get_center_point(self):
        return self.coords_to_point(0, 0)

    def get_x_unit_size(self):
        return self.get_x_axis().get_unit_size()

    def get_y_unit_size(self):
        return self.get_x_axis().get_unit_size()

    def get_axes(self):
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
