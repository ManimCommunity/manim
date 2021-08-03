from manim import *
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

__module_test__ = "coordinate_system"


@frames_comparison
def test_number_plane(scene):
    plane = NumberPlane(
        x_range=[-4, 6, 1],
        axis_config={"include_numbers": True},
        x_axis_config={"unit_size": 1.2},
        y_range=[-2, 5],
        y_length=6,
        y_axis_config={"label_direction": UL},
    )

    scene.add(plane)
