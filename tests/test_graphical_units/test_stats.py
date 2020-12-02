import pytest

from manim import *
from ..utils.testing_utils import get_scenes_to_test
from ..utils.GraphicalUnitTester import GraphicalUnitTester


class HistogramTest(GraphScene):
    def construct(self):
        self.setup_axes()
        a = Histogram([0, 1, 2, 3, 4], [5, 1, 4, 2, 3], self])
        self.play(Animation(a))


MODULE_NAME = "stats"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
