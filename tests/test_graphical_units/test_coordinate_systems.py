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


@frames_comparison
def test_line_graph(scene):
    plane = NumberPlane()
    first_line = plane.get_line_graph(
        x_values=[-3, 1],
        y_values=[-2, 2],
        line_color=YELLOW,
    )
    second_line = plane.get_line_graph(
        x_values=[0, 2, 2, 4],
        y_values=[0, 0, 2, 4],
        line_color=RED,
    )

    scene.add(plane, first_line, second_line)


@frames_comparison
def test_implicit_graph(scene):
    ax = Axes()
    graph = ax.get_implicit_curve(lambda x, y: x ** 2 + y ** 2 - 4)
    scene.add(ax, graph)
