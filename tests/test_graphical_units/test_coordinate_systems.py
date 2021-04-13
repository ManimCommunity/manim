import pytest

from manim import *

from ..utils.GraphicalUnitTester import GraphicalUnitTester
from ..utils.testing_utils import get_scenes_to_test


class NumberPlaneTest(Scene):
    def construct(self):
        plane = NumberPlane(
            x_range=[-4, 6, 1],
            axis_config={"include_numbers": True},
            x_axis_config={"unit_size": 1.2},
            y_range=[-2, 5],
            y_length=6,
            y_axis_config={"label_direction": UL},
        )
        plane.shift(1.5 * DL)
        self.play(Animation(plane))


MODULE_NAME = "coordinate_systems"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
