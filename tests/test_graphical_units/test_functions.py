from __future__ import annotations

import numpy as np

from manim.constants import PI
from manim.mobject.graphing.functions import FunctionGraph, ImplicitFunction
from manim.utils.color import BLUE
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "functions"


@frames_comparison
def test_FunctionGraph(scene):
    graph = FunctionGraph(lambda x: 2 * np.cos(0.5 * x), x_range=[-PI, PI], color=BLUE)
    scene.add(graph)


@frames_comparison
def test_ImplicitFunction(scene):
    graph = ImplicitFunction(lambda x, y: x**2 + y**2 - 9)
    scene.add(graph)
