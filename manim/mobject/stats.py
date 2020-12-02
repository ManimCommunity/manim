r"""Mobjects that have to do with statistics and statistical analysis.

Examples
--------
.. manim:: ShowHistogram
    :save_last_frame:

    class ShowHistogram(GraphScene):
        def construct(self):
            self.setup_axes()
            histogram = Histogram([0, 1, 2, 3, 4], [5, 1, 4, 2, 3], self)

            self.add(histogram)

"""

import numpy as np
from ..mobject.functions import ParametricFunction
from ..scene.graph_scene import GraphScene
from ..constants import *

__all__ = [
    "Histogram",
]


class Histogram(ParametricFunction):
    """A Histogram VMobject (optionally scaled to a graph) that can be used to
        easily show statistical data.

    Parameters
    ----------
    x : :class:`list`, :class:`tuple`, :class:`numpy.ndarray`
        The list, tuple, or array of X values for the Histogram to plot from.

    y : :class:`list`, :class:`tuple`, :class:`numpy.ndarray`
        The list, tuple, or array of Y values for the Histogram to plot from.

    graph_scene : Optional[:class:`.GraphScene`], optional
        The :class:`.GraphScene` instance with which to align the Histogram's points.
        When using Histogram in a GraphScene, this should be passed in as ``self``.

    Attributes
    ----------
    hist_points : :class:`list`
        A list of the points in the Histogram, ordered by X value. You cannot update
        this attribute.

    lines : :class:`list`
        A list of :class:`dict` with information for the lines connecting :attr:`.points`. You
            cannot update this attribute. Each dictionary's keys are the following:

        * ``"func"``: The function representing the line connecting two adjacent points.

        * ``"domain"``: The domain in which to use the function to draw the line (not
        including endpoints—those are determined from the points themselves)."""

    def __init__(self, x, y, graph_scene=None, **kwargs):
        if graph_scene is None:
            graph_scene = GraphScene()
            graph_scene.setup()
            graph_scene.setup_axes()
            graph_scene.remove(graph_scene.x_axis, graph_scene.y_axis)

        if "t_min" not in kwargs.keys():
            if "x_min" in kwargs.keys():
                kwargs["t_min"] = kwargs["x_min"]

            else:
                kwargs["t_min"] = min(x)

        if "t_max" not in kwargs.keys():
            if "x_max" in kwargs.keys():
                kwargs["t_max"] = kwargs["x_max"]

            else:
                kwargs["t_max"] = max(x)

        self.kwargs = kwargs
        self.x_axis = graph_scene.x_axis
        self.y_axis = graph_scene.y_axis
        self._hist_points = sorted(list(zip(x, y)), key=lambda item: item[0])
        self._lines = []

        for point1, point2 in zip(self._hist_points[:-1], self._hist_points[1:]):
            self._lines.append(
                {
                    "func": self._get_connecting_function(point1, point2),
                    "domain": (point1[0], point2[0]),
                }
            )

        def function(t):
            if (
                t < self._hist_points[0][0]
                or t > self._hist_points[len(self._hist_points) - 1][0]
            ):
                return self._coords_to_point(t, np.nan)

            for x, y in self._hist_points:
                if t == x:
                    return self._coords_to_point(x, y)

            for line in self._lines:
                if t > line["domain"][0] and t < line["domain"][1]:
                    return self._coords_to_point(t, line["func"](t))

        super().__init__(function, **kwargs)

    def _coords_to_point(self, x, y):
        assert hasattr(self, "x_axis") and hasattr(self, "y_axis")
        result = self.x_axis.number_to_point(x)[0] * RIGHT
        result += self.y_axis.number_to_point(y)[1] * UP
        return result

    def _get_connecting_function(self, point1, point2):
        return (
            lambda x: ((point2[1] - point1[1]) / (point2[0] - point1[0])) * x
            + point1[1]
            - ((point2[1] - point1[1]) / (point2[0] - point1[0])) * point1[0]
        )

    @property
    def lines(self):
        return self._lines

    @property
    def hist_points(self):
        return self._hist_points
