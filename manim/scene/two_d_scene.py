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

__all__ = ["TwoDScene"]

from manim import Write
from .. import config
from ..constants import *
from ..mobject.coordinate_systems import NumberPlane
from ..mobject.functions import ParametricFunction
from ..mobject.geometry import Line
from ..mobject.geometry import Rectangle
from ..mobject.svg.tex_mobject import MathTex
from ..mobject.types.vectorized_mobject import VGroup
from ..mobject.types.vectorized_mobject import VectorizedPoint
from ..scene.scene import Scene
from ..utils.color import color_gradient, BLACK, LIGHT_GREY
from ..utils.color import invert_color
from ..utils.config_ops import merge_dicts_recursively


class TwoDScene(Scene,NumberPlane):
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
                "x_max": 8,
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

    def setup_axes(self, animate=False,x_vals=None,y_vals=None,run_time=1):
        self.get_y_axis().shift(self.center_point) # Replace with self.shift(self.center_point) once this bug is fixed!
        self.get_x_axis().shift(self.center_point)

        if animate:
            self.play(Write(self.axes,run_time=run_time))
        else:
            self.add(*self.axes)


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
