import pytest

from manim import *
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

from ..utils.GraphicalUnitTester import GraphicalUnitTester
from ..utils.testing_utils import get_scenes_to_test

__module_test__ = "plot"


@frames_comparison
def test_axes(scene):
    graph = Axes(
        x_range=[-10, 10, 1],
        y_range=[-10, 10, 1],
        x_length=6,
        y_length=6,
        color=WHITE,
        axis_config={"exclude_origin_tick": False},
    )
    labels = graph.get_axis_labels()
    scene.add(graph, labels)
