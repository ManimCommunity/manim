import numpy as np
import pytest

from manim import *

from ..utils.GraphicalUnitTester import GraphicalUnitTester
from ..utils.testing_utils import get_scenes_to_test


class FunctionGraphTest(Scene):
    def construct(self):
        graph = FunctionGraph(
            lambda x: 2 * np.cos(0.5 * x), x_range=[-PI, PI], color=BLUE
        )
        self.add(graph)


MODULE_NAME = "functions"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
