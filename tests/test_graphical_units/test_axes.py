from manim import *
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

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


@frames_comparison()
def test_plot_functions(scene):
    ax = Axes(x_range=(-10, 10.3), y_range=(-1.5, 1.5))
    graph = ax.get_graph(lambda x: x ** 2)
    scene.add(ax, graph)


@frames_comparison
def test_custom_coordinates(scene):
    ax = Axes(x_range=[0, 10])

    ax.add_coordinates(
        dict(zip([x for x in range(1, 10)], [Tex("str") for _ in range(1, 10)]))
    )
    scene.add(ax)
