"""A scene for plotting / graphing functions.

Examples
--------

.. manim:: FunctionPlotWithLabbeledYAxis
    :save_last_frame:

    class FunctionPlotWithLabbeledYAxis(GraphScene):
        CONFIG = {
            "y_min": 0,
            "y_max": 100,
            "y_axis_config": {"tick_frequency": 10},
            "y_labeled_nums": np.arange(0, 100, 10)
        }

        def construct(self):
            self.setup_axes()
            dot = Dot().move_to(self.coords_to_point(PI / 2, 20))
            func_graph = self.get_graph(lambda x: 20 * np.sin(x))
            self.add(dot,func_graph)


.. manim:: GaussianFunctionPlot
    :save_last_frame:

    amp = 5
    mu = 3
    sig = 1

    def gaussian(x):
        return amp * np.exp((-1 / 2 * ((x - mu) / sig) ** 2))

    class GaussianFunctionPlot(GraphScene):
        def construct(self):
            self.setup_axes()
            graph = self.get_graph(gaussian, x_min=-1, x_max=10)
            graph.set_stroke(width=5)
            self.add(graph)

"""

__all__ = ["GraphScene"]

from manim import Write

from ..mobject.coordinate_systems import NumberPlane
from ..utils.config_ops import merge_dicts_recursively

from .. import config
from ..animation.creation import ShowCreation
from ..constants import *
from ..mobject.functions import ParametricFunction
from ..mobject.geometry import Line
from ..mobject.geometry import Rectangle
from ..mobject.svg.tex_mobject import MathTex
from ..mobject.types.vectorized_mobject import VGroup
from ..mobject.types.vectorized_mobject import VectorizedPoint
from ..scene.scene import Scene
from ..utils.color import color_gradient, BLUE, GREEN, YELLOW, BLACK, WHITE, LIGHT_GREY
from ..utils.color import invert_color
from ..utils.space_ops import angle_of_vector

class GraphScene(Scene,NumberPlane):
    CONFIG = {
        "center_point": 2.5 * DOWN + 4 * LEFT,
        "axis_config": {
            "stroke_color": LIGHT_GREY,
            "stroke_width": 2,
            "include_ticks": True,
            "line_to_number_buff": 0.2,
            "label_direction": DOWN,
            "number_scale_val": 0.8,
            "color": LIGHT_GREY,
            "exclude_zero_from_default_numbers": True,
        },
        "x_axis_config": {
                "x_min": -1,
                "x_max": 5,
                "unit_size": 1,
                "color": LIGHT_GREY,
                "tick_frequency": 1,
                       },
        "y_axis_config": {
            "label_direction": LEFT,
            "x_min": -1,
            "x_max": 10,
            "unit_size": 1,
            "color": LIGHT_GREY,
            "tick_frequency": 1,
        },
    }

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        NumberPlane.__init__(self,**merge_dicts_recursively(self.CONFIG,kwargs))
     #   self.mobjects = [] # Get rid of immediately displayed axes from Axes.__init__()

    def setup_axes(self, animate=False,x_vals=None,y_vals=None,run_time=1):
        self.get_y_axis().shift(self.center_point) # Replace with self.shift(self.center_point) once this bug is fixed!
        self.get_x_axis().shift(self.center_point)

        default_coord_labels = self.get_coordinate_labels(x_vals, y_vals)  # Vgroup(xm,ym)
        if animate:
            self.play(Write(self.axes,run_time=run_time),
                      Write(default_coord_labels,run_time=run_time))
        else:
            self.add(*self.axes,default_coord_labels)

    def get_coordinate_labels(self, x_vals=None, y_vals=None):
        default_labels = super().get_coordinate_labels(x_vals,y_vals)
        if x_vals == y_vals is None:
            default_labels[0].remove(default_labels[0][0])
            default_labels[1].remove(default_labels[1][0])
        return default_labels

    def get_graph_label(
        self,
        graph,
        label="f(x)",
        x_val=None,
        direction=RIGHT,
        buff=MED_SMALL_BUFF,
        color=None,
    ):
        """
        This method returns a properly positioned label for the passed graph,
        styled with the passed parameters.

        Parameters
        ----------
        graph : ParametricFunction
            The curve of the function plotted.

        label : str, optional
            The label for the function's curve.

        x_val : int, float, optional
            The x_value with which the label should be aligned.

        direction : np.ndarray, list, tuple
            The cartesian position, relative to the curve that the label will be at.
            e.g LEFT, RIGHT

        buff : float, int, option
            The buffer space between the curve and the label

        color : str, optional
            The color of the label.

        Returns
        -------
        :class:`~.MathTex`
            The LaTeX of the passed 'label' parameter

        """
        label = MathTex(label)
        color = color or graph.get_color()
        label.set_color(color)
        if x_val is None:
            # Search from right to left
            for x in np.linspace(self.x_max, self.x_min, 100):
                point = self.input_to_graph_point(x, graph)
                if point[1] < config["frame_y_radius"]:
                    break
            x_val = x
        label.next_to(self.input_to_graph_point(x_val, graph), direction, buff=buff)
        label.shift_onto_screen()
        return label

    def get_riemann_rectangles(
        self,
        graph,
        x_min=None,
        x_max=None,
        dx=0.1,
        input_sample_type="left",
        bounded_graph=None,
        stroke_width=1,
        stroke_color=BLACK,
        fill_opacity=1,
        start_color=None,
        end_color=None,
        show_signed_area=True,
        width_scale_factor=1.001,
    ):
        """
        This method returns the VGroup() of the Riemann Rectangles for
        a particular curve.

        Parameters
        ----------
        graph : ParametricFunction
            The graph whose area needs to be approximated
            by the Riemann Rectangles.

        x_min : int, float, optional
            The lower bound from which to start adding rectangles

        x_max : int, float, optional
            The upper bound where the rectangles stop.

        dx : int, float, optional
            The smallest change in x-values that is
            considered significant.

        input_sample_type : {"left", "right", "center"}
            Can be any of "left", "right" or "center

        stroke_width : int, float, optional
            The stroke_width of the border of the rectangles.

        stroke_color : str, optional
            The string of hex colour of the rectangle's border.

        fill_opacity : int, float
            The opacity of the rectangles. Takes values from 0 to 1.

        start_color : str, optional
            The hex starting colour for the rectangles,
            this will, if end_color is a different colour,
            make a nice gradient.

        end_color : str, optional
            The hex ending colour for the rectangles,
            this will, if start_color is a different colour,
            make a nice gradient.

        show_signed_area : bool, optional
            Whether or not to indicate -ve area if curve dips below
            x-axis.

        width_scale_factor : int, float, optional
            How much the width of the rectangles are scaled by when transforming.

        Returns
        -------
        VGroup
            A VGroup containing the Riemann Rectangles.

        """
        x_min = x_min if x_min is not None else self.x_min
        x_max = x_max if x_max is not None else self.x_max
        if start_color is None:
            start_color = self.default_riemann_start_color
        if end_color is None:
            end_color = self.default_riemann_end_color
        rectangles = VGroup()
        x_range = np.arange(x_min, x_max, dx)
        colors = color_gradient([start_color, end_color], len(x_range))
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
            if bounded_graph == None:
                y_point = 0
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

            rect = Rectangle()
            rect.replace(points, stretch=True)
            if graph_point[1] < self.graph_origin[1] and show_signed_area:
                fill_color = invert_color(color)
            else:
                fill_color = color
            rect.set_fill(fill_color, opacity=fill_opacity)
            rect.set_stroke(stroke_color, width=stroke_width)
            rectangles.add(rect)
        return rectangles

    def get_riemann_rectangles_list(
        self, graph, n_iterations, max_dx=0.5, power_base=2, stroke_width=1, **kwargs
    ):
        """
        This method returns a list of multiple VGroups of Riemann
        Rectangles. The inital VGroups are relatively inaccurate,
        but the closer you get to the end the more accurate the Riemann
        rectangles become

        Parameters
        ----------
        graph : ParametricFunction
            The graph whose area needs to be approximated
            by the Riemann Rectangles.

        n_iterations : int,
            The number of VGroups of successive accuracy that are needed.

        max_dx : int, float, optional
            The maximum change in x between two VGroups of Riemann Rectangles

        power_base : int, float, optional
            Defaults to 2

        stroke_width : int, float, optional
            The stroke_width of the border of the rectangles.

        **kwargs
            Any valid keyword arguments of get_riemann_rectangles.

        Returns
        -------
        list
            The list of Riemann Rectangles of increasing accuracy.
        """
        return [
            self.get_riemann_rectangles(
                graph=graph,
                dx=float(max_dx) / (power_base ** n),
                stroke_width=float(stroke_width) / (power_base ** n),
                **kwargs,
            )
            for n in range(n_iterations)
        ]

    def get_vertical_line_to_graph(self, x, graph, line_class=Line, **line_kwargs):
        """
        This method returns a Vertical line from the x-axis to
        the corresponding point on the graph/curve.

        Parameters
        ----------
        x : int, float
            The x-value at which the line should be placed/calculated.

        graph : ParametricFunction
            The graph on which the line should extend to.

        line_class : Line and similar
            The type of line that should be used.
            Defaults to Line.

        **line_kwargs
            Any valid keyword arguments of the object passed in "line_class"
            If line_class is Line, any valid keyword arguments of Line are allowed.

        Return
        ------
        An object of type passed in "line_class"
            Defaults to Line
        """
        if "color" not in line_kwargs:
            line_kwargs["color"] = graph.get_color()
        return line_class(
            self.coords_to_point(x, 0),
            self.input_to_graph_point(x, graph),
            **line_kwargs,
        )

    def get_vertical_lines_to_graph(
        self, graph, x_min=None, x_max=None, num_lines=20, **kwargs
    ):
        """
        Obtains multiple lines from the x axis to the Graph/curve.

        Parameters
        ----------
        graph : ParametricFunction
            The graph on which the line should extend to.

        x_min : int, float, optional
            The lower bound from which lines can appear.

        x_max : int, float, optional
            The upper bound until which the lines can appear.

        num_lines : int, optional
            The number of lines (evenly spaced)
            that are needed.

        Returns
        -------
        VGroup
            The VGroup of the evenly spaced lines.

        """
        x_min = x_min or self.x_min
        x_max = x_max or self.x_max
        return VGroup(
            *[
                self.get_vertical_line_to_graph(x, graph, **kwargs)
                for x in np.linspace(x_min, x_max, num_lines)
            ]
        )
