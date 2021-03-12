import pytest

from manim import *
from ..utils.testing_utils import get_scenes_to_test
from ..utils.GraphicalUnitTester import GraphicalUnitTester


class NumberPlaneTest(Scene):
    def construct(self):
        plane = NumberPlane(
            axis_config={"include_numbers": True},
            y_axis_config={"x_min": -2, "x_max": 5, "width": 6, "label_direction": UL},
            x_axis_config={"x_min": -4, "x_max": 6, "unit_size": 1.2},
            center_point=2 * DL,
        )
        self.play(Animation(plane))


MODULE_NAME = "coordinate_systems"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
