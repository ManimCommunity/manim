from __future__ import annotations

from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "plot"


@frames_comparison
def test_axes(scene):
    graph = Axes(
        x_range=[-10, 10, 1],
        y_range=[-10, 10, 1],
        x_length=6,
        y_length=6,
        color=WHITE,
    )
    labels = graph.get_axis_labels()
    scene.add(graph, labels)


@frames_comparison
def test_plot_functions(scene, use_vectorized):
    ax = Axes(x_range=(-10, 10.3), y_range=(-1.5, 1.5))
    graph = ax.plot(lambda x: x**2, use_vectorized=use_vectorized)
    scene.add(ax, graph)


@frames_comparison
def test_custom_coordinates(scene):
    ax = Axes(x_range=[0, 10])

    ax.add_coordinates(
        dict(zip(list(range(1, 10)), [Tex("str") for _ in range(1, 10)])),
    )
    scene.add(ax)


@frames_comparison
def test_get_axis_labels(scene):
    ax = Axes()
    labels = ax.get_axis_labels(Tex("$x$-axis").scale(0.7), Tex("$y$-axis").scale(0.45))
    scene.add(ax, labels)


@frames_comparison
def test_get_x_axis_label(scene):
    ax = Axes(x_range=(0, 8), y_range=(0, 5), x_length=8, y_length=5)
    x_label = ax.get_x_axis_label(
        Tex("$x$-values").scale(0.65),
        edge=DOWN,
        direction=DOWN,
        buff=0.5,
    )
    scene.add(ax, x_label)


@frames_comparison
def test_get_y_axis_label(scene):
    ax = Axes(x_range=(0, 8), y_range=(0, 5), x_length=8, y_length=5)
    y_label = ax.get_y_axis_label(
        Tex("$y$-values").scale(0.65).rotate(90 * DEGREES),
        edge=LEFT,
        direction=LEFT,
        buff=0.3,
    )
    scene.add(ax, y_label)


@frames_comparison
def test_axis_tip_default_width_height(scene):
    ax = Axes(
        x_range=(0, 4),
        y_range=(0, 4),
        axis_config={"include_numbers": True, "include_tip": True},
    )

    scene.add(ax)


@frames_comparison
def test_axis_tip_custom_width_height(scene):
    ax = Axes(
        x_range=(0, 4),
        y_range=(0, 4),
        axis_config={"include_numbers": True, "include_tip": True},
        x_axis_config={"tip_width": 1, "tip_height": 0.1},
        y_axis_config={"tip_width": 0.1, "tip_height": 1},
    )

    scene.add(ax)


@frames_comparison
def test_plot_derivative_graph(scene, use_vectorized):
    ax = NumberPlane(y_range=[-1, 7], background_line_style={"stroke_opacity": 0.4})

    curve_1 = ax.plot(lambda x: x**2, color=PURPLE_B, use_vectorized=use_vectorized)
    curve_2 = ax.plot_derivative_graph(curve_1, use_vectorized=use_vectorized)
    curve_3 = ax.plot_antiderivative_graph(curve_1, use_vectorized=use_vectorized)
    curves = VGroup(curve_1, curve_2, curve_3)
    scene.add(ax, curves)


@frames_comparison
def test_plot(scene, use_vectorized):
    # construct the axes
    ax_1 = Axes(
        x_range=[0.001, 6],
        y_range=[-8, 2],
        x_length=5,
        y_length=3,
        tips=False,
    )
    ax_2 = ax_1.copy()
    ax_3 = ax_1.copy()

    # position the axes
    ax_1.to_corner(UL)
    ax_2.to_corner(UR)
    ax_3.to_edge(DOWN)
    axes = VGroup(ax_1, ax_2, ax_3)

    # create the logarithmic curves
    def log_func(x):
        return np.log(x)

    # a curve without adjustments; poor interpolation.
    curve_1 = ax_1.plot(log_func, color=PURE_RED, use_vectorized=use_vectorized)

    # disabling interpolation makes the graph look choppy as not enough
    # inputs are available
    curve_2 = ax_2.plot(
        log_func, use_smoothing=False, color=ORANGE, use_vectorized=use_vectorized
    )

    # taking more inputs of the curve by specifying a step for the
    # x_range yields expected results, but increases rendering time.
    curve_3 = ax_3.plot(
        log_func,
        x_range=(0.001, 6, 0.001),
        color=PURE_GREEN,
        use_vectorized=use_vectorized,
    )

    curves = VGroup(curve_1, curve_2, curve_3)

    scene.add(axes, curves)


@frames_comparison
def test_get_graph_label(scene):
    ax = Axes()
    sin = ax.plot(lambda x: np.sin(x), color=PURPLE_B)
    label = ax.get_graph_label(
        graph=sin,
        label=MathTex(r"\frac{\pi}{2}"),
        x_val=PI / 2,
        dot=True,
        dot_config={"radius": 0.04},
        direction=UR,
    )

    scene.add(ax, sin, label)


@frames_comparison
def test_get_lines_to_point(scene):
    ax = Axes()
    circ = Circle(radius=0.5).move_to([-4, -1.5, 0])

    lines_1 = ax.get_lines_to_point(circ.get_right(), color=GREEN_B)
    lines_2 = ax.get_lines_to_point(circ.get_corner(DL), color=BLUE_B)
    scene.add(ax, lines_1, lines_2, circ)


@frames_comparison
def test_plot_line_graph(scene):
    plane = NumberPlane(
        x_range=(0, 7),
        y_range=(0, 5),
        x_length=7,
        axis_config={"include_numbers": True},
    )

    line_graph = plane.plot_line_graph(
        x_values=[0, 1.5, 2, 2.8, 4, 6.25],
        y_values=[1, 3, 2.25, 4, 2.5, 1.75],
        line_color=GOLD_E,
        vertex_dot_style={"stroke_width": 3, "fill_color": PURPLE},
        vertex_dot_radius=0.04,
        stroke_width=4,
    )
    # test that the line and dots can be accessed afterwards
    line_graph["line_graph"].set_stroke(width=2)
    line_graph["vertex_dots"].scale(2)
    scene.add(plane, line_graph)


@frames_comparison
def test_t_label(scene):
    # defines the axes and linear function
    axes = Axes(x_range=[-1, 10], y_range=[-1, 10], x_length=9, y_length=6)
    func = axes.plot(lambda x: x, color=BLUE)
    # creates the T_label
    t_label = axes.get_T_label(x_val=4, graph=func, label=Tex("$x$-value"))
    scene.add(axes, func, t_label)


@frames_comparison
def test_get_area(scene):
    ax = Axes().add_coordinates()
    curve1 = ax.plot(
        lambda x: 2 * np.sin(x),
        x_range=[-5, ax.x_range[1]],
        color=DARK_BLUE,
    )
    curve2 = ax.plot(lambda x: (x + 4) ** 2 - 2, x_range=[-5, -2], color=RED)
    area1 = ax.get_area(
        curve1,
        x_range=(PI / 2, 3 * PI / 2),
        color=(GREEN_B, GREEN_D),
        opacity=1,
    )
    area2 = ax.get_area(
        curve1,
        x_range=(-4.5, -2),
        color=(RED, YELLOW),
        opacity=0.2,
        bounded_graph=curve2,
    )

    scene.add(ax, curve1, curve2, area1, area2)


@frames_comparison
def test_get_area_with_boundary_and_few_plot_points(scene):
    ax = Axes(x_range=[-2, 2], y_range=[-2, 2], color=WHITE)
    f1 = ax.plot(lambda t: t, [-1, 1, 0.5])
    f2 = ax.plot(lambda t: 1, [-1, 1, 0.5])
    a1 = ax.get_area(f1, [-1, 0.75], color=RED)
    a2 = ax.get_area(f1, [-0.75, 1], bounded_graph=f2, color=GREEN)

    scene.add(ax, f1, f2, a1, a2)


@frames_comparison
def test_get_riemann_rectangles(scene, use_vectorized):
    ax = Axes(y_range=[-2, 10])
    quadratic = ax.plot(lambda x: 0.5 * x**2 - 0.5, use_vectorized=use_vectorized)

    # the rectangles are constructed from their top right corner.
    # passing an iterable to `color` produces a gradient
    rects_right = ax.get_riemann_rectangles(
        quadratic,
        x_range=[-4, -3],
        dx=0.25,
        color=(TEAL, BLUE_B, DARK_BLUE),
        input_sample_type="right",
    )

    # the colour of rectangles below the x-axis is inverted
    # due to show_signed_area
    rects_left = ax.get_riemann_rectangles(
        quadratic,
        x_range=[-1.5, 1.5],
        dx=0.15,
        color=YELLOW,
    )

    bounding_line = ax.plot(lambda x: 1.5 * x, color=BLUE_B, x_range=[3.3, 6])
    bounded_rects = ax.get_riemann_rectangles(
        bounding_line,
        bounded_graph=quadratic,
        dx=0.15,
        x_range=[4, 5],
        show_signed_area=False,
        color=(MAROON_A, RED_B, PURPLE_D),
    )

    scene.add(ax, bounding_line, quadratic, rects_right, rects_left, bounded_rects)


@frames_comparison(base_scene=ThreeDScene)
def test_get_z_axis_label(scene):
    ax = ThreeDAxes()
    lab = ax.get_z_axis_label(Tex("$z$-label"))
    scene.set_camera_orientation(phi=2 * PI / 5, theta=PI / 5)
    scene.add(ax, lab)


@frames_comparison
def test_polar_graph(scene):
    polar = PolarPlane()
    r = lambda theta: 4 * np.sin(theta * 4)
    polar_graph = polar.plot_polar_graph(r)
    scene.add(polar, polar_graph)


@frames_comparison
def test_log_scaling_graph(scene):
    ax = Axes(
        x_range=[0, 8],
        y_range=[-2, 4],
        x_length=10,
        y_axis_config={"scaling": LogBase()},
    )
    ax.add_coordinates()

    gr = ax.plot(lambda x: x, use_smoothing=False, x_range=[0.01, 8])

    scene.add(ax, gr)
