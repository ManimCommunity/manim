"""Mobjects that represent coordinate systems."""

__all__ = ["CoordinateSystem", "Axes", "ThreeDAxes", "NumberPlane", "ComplexPlane"]


import math
import numbers

import numpy as np

from .. import config
from ..constants import *
from ..mobject.functions import ParametricFunction
from ..mobject.geometry import Arrow, DashedLine, Line
from ..mobject.number_line import NumberLine
from ..mobject.svg.tex_mobject import MathTex
from ..mobject.types.vectorized_mobject import VGroup
from ..utils.color import BLUE, BLUE_D, LIGHT_GREY, WHITE
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

        self.x_range = np.array(x_range)
        self.y_range = np.array(y_range)
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
    def __init__(
        self,
        x_range=None,
        y_range=None,
        x_length=config.frame_width - 2,
        y_length=config.frame_height - 2,
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
        new_config = merge_dicts_recursively(self.axis_config, axis_config)
        new_config["length"] = length
        axis = NumberLine(range_terms, **new_config)

        # without the if/elif, graph does not exist when min > 0 or max < 0
        if range_terms[0] > 0:
            axis.shift(-axis.number_to_point(range_terms[0]))
        elif range_terms[1] < 0:
            axis.shift(-axis.number_to_point(range_terms[1]))
        else:
            axis.shift(-axis.number_to_point(0))
        return axis

    def coords_to_point(self, *coords):
        origin = self.x_axis.number_to_point(0)
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


class ThreeDAxes(Axes):
    def __init__(
        self,
        x_range=None,
        y_range=None,
        z_range=None,
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
        x_range = np.array([-5.5, 5.5, 1]) if x_range is None else x_range
        y_range = np.array([-5.5, 5.5, 1]) if y_range is None else y_range
        z_range = np.array([-3.5, 3.5, 1]) if z_range is None else z_range

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
        z_axis.shift(self.x_axis.n2p(0))

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
    def __init__(
        self,
        x_range=None,
        y_range=None,
        x_length=config.frame_width,
        y_length=config.frame_height,
        axis_config=None,
        y_axis_config=None,
        background_line_style=None,
        faded_line_style=None,
        faded_line_ratio=1,
        make_smooth_after_applying_functions=True,
        **kwargs,
    ):

        # use ceilings and floors to make the box bound the screen when None is passed.
        if x_range is None:
            x_range = [
                math.ceil(-config["frame_x_radius"]),
                math.floor(config["frame_x_radius"]),
                1.0,
            ]
        if y_range is None:
            y_range = [
                math.ceil(-config["frame_y_radius"]),
                math.floor(config["frame_y_radius"]),
                1.0,
            ]

        y_range = np.array(y_range)
        x_range = np.array(x_range)

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
        inputs = np.arange(
            axis_perpendicular_to.x_min, axis_perpendicular_to.x_max + step, step
        )
        for k, x in enumerate(inputs):
            new_line = line.copy()
            new_line.shift(axis_perpendicular_to.n2p(x) - axis_perpendicular_to.n2p(0))
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
