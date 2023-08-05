from __future__ import annotations

from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

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
    first_line = plane.plot_line_graph(
        x_values=[-3, 1],
        y_values=[-2, 2],
        line_color=YELLOW,
    )
    second_line = plane.plot_line_graph(
        x_values=[0, 2, 2, 4],
        y_values=[0, 0, 2, 4],
        line_color=RED,
    )

    scene.add(plane, first_line, second_line)


@frames_comparison(base_scene=ThreeDScene)
def test_plot_surface(scene):
    axes = ThreeDAxes(x_range=(-5, 5, 1), y_range=(-5, 5, 1), z_range=(-5, 5, 1))

    def param_trig(u, v):
        x = u
        y = v
        z = 2 * np.sin(x) + 2 * np.cos(y)
        return z

    trig_plane = axes.plot_surface(
        param_trig,
        u_range=(-5, 5),
        v_range=(-5, 5),
        color=BLUE,
    )

    scene.add(axes, trig_plane)


@frames_comparison(base_scene=ThreeDScene)
def test_plot_surface_colorscale(scene):
    axes = ThreeDAxes(x_range=(-3, 3, 1), y_range=(-3, 3, 1), z_range=(-5, 5, 1))

    def param_trig(u, v):
        x = u
        y = v
        z = 2 * np.sin(x) + 2 * np.cos(y)
        return z

    trig_plane = axes.plot_surface(
        param_trig,
        u_range=(-3, 3),
        v_range=(-3, 3),
        colorscale=[BLUE, GREEN, YELLOW, ORANGE, RED],
    )

    scene.add(axes, trig_plane)


@frames_comparison
def test_implicit_graph(scene):
    ax = Axes()
    graph = ax.plot_implicit_curve(lambda x, y: x**2 + y**2 - 4)
    scene.add(ax, graph)


@frames_comparison
def test_plot_log_x_axis(scene):
    ax = Axes(
        x_range=[-1, 4],
        y_range=[0, 3],
        x_axis_config={"scaling": LogBase()},
    )

    graph = ax.plot(lambda x: 2 if x < 10 else 1, x_range=[-1, 4])
    scene.add(ax, graph)


@frames_comparison
def test_plot_log_x_axis_vectorized(scene):
    ax = Axes(
        x_range=[-1, 4],
        y_range=[0, 3],
        x_axis_config={"scaling": LogBase()},
    )

    graph = ax.plot(
        lambda x: np.where(x < 10, 2, 1), x_range=[-1, 4], use_vectorized=True
    )
    scene.add(ax, graph)


@frames_comparison
def test_number_plane_log(scene):
    """Test that NumberPlane generates its lines properly with a LogBase"""
    # y_axis log
    plane1 = (
        NumberPlane(
            x_range=[0, 8, 1],
            y_range=[-2, 5],
            y_length=6,
            x_length=10,
            y_axis_config={"scaling": LogBase()},
        )
        .add_coordinates()
        .scale(1 / 2)
    )

    # x_axis log
    plane2 = (
        NumberPlane(
            x_range=[0, 8, 1],
            y_range=[-2, 5],
            y_length=6,
            x_length=10,
            x_axis_config={"scaling": LogBase()},
            faded_line_ratio=4,
        )
        .add_coordinates()
        .scale(1 / 2)
    )

    scene.add(VGroup(plane1, plane2).arrange())


@frames_comparison
def test_gradient_line_graph_x_axis(scene):
    """Test that using `colorscale` generates a line whose gradient matches the y-axis"""
    axes = Axes(x_range=[-3, 3], y_range=[-3, 3])

    curve = axes.plot(
        lambda x: 0.1 * x**3,
        x_range=(-3, 3, 0.001),
        colorscale=[BLUE, GREEN, YELLOW, ORANGE, RED],
        colorscale_axis=0,
    )

    scene.add(axes, curve)


@frames_comparison
def test_gradient_line_graph_y_axis(scene):
    """Test that using `colorscale` generates a line whose gradient matches the y-axis"""
    axes = Axes(x_range=[-3, 3], y_range=[-3, 3])

    curve = axes.plot(
        lambda x: 0.1 * x**3,
        x_range=(-3, 3, 0.001),
        colorscale=[BLUE, GREEN, YELLOW, ORANGE, RED],
        colorscale_axis=1,
    )

    scene.add(axes, curve)
