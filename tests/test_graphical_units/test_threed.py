from __future__ import annotations

from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "threed"


@frames_comparison(base_scene=ThreeDScene)
def test_AddFixedInFrameMobjects(scene):
    scene.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
    text = Tex("This is a 3D tex")
    scene.add_fixed_in_frame_mobjects(text)


@frames_comparison(base_scene=ThreeDScene)
def test_Cube(scene):
    scene.add(Cube())


@frames_comparison(base_scene=ThreeDScene)
def test_Sphere(scene):
    scene.add(Sphere())


@frames_comparison(base_scene=ThreeDScene)
def test_Dot3D(scene):
    scene.add(Dot3D())


@frames_comparison(base_scene=ThreeDScene)
def test_Cone(scene):
    scene.add(Cone(resolution=16))


def test_Cone_get_start_and_get_end():
    cone = Cone().shift(RIGHT).rotate(PI / 4, about_point=ORIGIN, about_edge=OUT)
    start = [0.70710678, 0.70710678, -1.0]
    end = [0.70710678, 0.70710678, 0.0]
    assert np.allclose(cone.get_start(), start, atol=0.01), (
        "start points of Cone do not match"
    )
    assert np.allclose(cone.get_end(), end, atol=0.01), (
        "end points of Cone do not match"
    )


@frames_comparison(base_scene=ThreeDScene)
def test_Cylinder(scene):
    scene.add(Cylinder())


@frames_comparison(base_scene=ThreeDScene)
def test_Line3D(scene):
    line1 = Line3D(resolution=16).shift(LEFT * 2)
    line2 = Line3D(resolution=16).shift(RIGHT * 2)
    perp_line = Line3D.perpendicular_to(line1, UP + OUT, resolution=16)
    parallel_line = Line3D.parallel_to(line2, DOWN + IN, resolution=16)
    scene.add(line1, line2, perp_line, parallel_line)


@frames_comparison(base_scene=ThreeDScene)
def test_Arrow3D(scene):
    scene.add(Arrow3D(resolution=16))


@frames_comparison(base_scene=ThreeDScene)
def test_Torus(scene):
    scene.add(Torus())


@frames_comparison(base_scene=ThreeDScene)
def test_Axes(scene):
    scene.add(ThreeDAxes())


@frames_comparison(base_scene=ThreeDScene)
def test_CameraMoveAxes(scene):
    """Tests camera movement to explore varied views of a static scene."""
    axes = ThreeDAxes()
    scene.add(axes)
    scene.add(Dot([1, 2, 3]))
    scene.move_camera(phi=PI / 8, theta=-PI / 8, frame_center=[1, 2, 3], zoom=2)


@frames_comparison(base_scene=ThreeDScene)
def test_CameraMove(scene):
    cube = Cube()
    scene.add(cube)
    scene.move_camera(phi=PI / 4, theta=PI / 4, frame_center=[0, 0, -1], zoom=0.5)


@frames_comparison(base_scene=ThreeDScene)
def test_AmbientCameraMove(scene):
    cube = Cube()
    scene.begin_ambient_camera_rotation(rate=0.5)
    scene.add(cube)
    scene.wait()


@frames_comparison(base_scene=ThreeDScene)
def test_MovingVertices(scene):
    scene.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
    vertices = [1, 2, 3, 4]
    edges = [(1, 2), (2, 3), (3, 4), (1, 3), (1, 4)]
    g = Graph(vertices, edges)
    scene.add(g)
    scene.play(
        g[1].animate.move_to([1, 1, 1]),
        g[2].animate.move_to([-1, 1, 2]),
        g[3].animate.move_to([1, -1, -1]),
        g[4].animate.move_to([-1, -1, 0]),
    )
    scene.wait()


@frames_comparison(base_scene=ThreeDScene)
def test_SurfaceColorscale(scene):
    resolution_fa = 16
    scene.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)
    axes = ThreeDAxes(x_range=(-3, 3, 1), y_range=(-3, 3, 1), z_range=(-4, 4, 1))

    def param_trig(u, v):
        x = u
        y = v
        z = y**2 / 2 - x**2 / 2
        return z

    trig_plane = Surface(
        lambda x, y: axes.c2p(x, y, param_trig(x, y)),
        resolution=(resolution_fa, resolution_fa),
        v_range=[-3, 3],
        u_range=[-3, 3],
    )
    trig_plane.set_fill_by_value(
        axes=axes, colorscale=[BLUE, GREEN, YELLOW, ORANGE, RED]
    )
    scene.add(axes, trig_plane)


@frames_comparison(base_scene=ThreeDScene)
def test_Y_Direction(scene):
    resolution_fa = 16
    scene.set_camera_orientation(phi=75 * DEGREES, theta=-120 * DEGREES)
    axes = ThreeDAxes(x_range=(0, 5, 1), y_range=(0, 5, 1), z_range=(-1, 1, 0.5))

    def param_surface(u, v):
        x = u
        y = v
        z = np.sin(x) * np.cos(y)
        return z

    surface_plane = Surface(
        lambda u, v: axes.c2p(u, v, param_surface(u, v)),
        resolution=(resolution_fa, resolution_fa),
        v_range=[0, 5],
        u_range=[0, 5],
    )
    surface_plane.set_style(fill_opacity=1)
    surface_plane.set_fill_by_value(
        axes=axes, colorscale=[(RED, -0.4), (YELLOW, 0), (GREEN, 0.4)], axis=1
    )
    scene.add(axes, surface_plane)


def test_get_start_and_end_Arrow3d():
    start, end = ORIGIN, np.array([2, 1, 0], dtype=np.float64)
    arrow = Arrow3D(start, end)
    assert np.allclose(arrow.get_start(), start, atol=0.01), (
        "start points of Arrow3D do not match"
    )
    assert np.allclose(arrow.get_end(), end, atol=0.01), (
        "end points of Arrow3D do not match"
    )


def test_type_conversion_in_Line3D():
    start, end = [0, 0, 0], [1, 1, 1]
    line = Line3D(start, end)
    type_table = [type(item) for item in [*line.get_start(), *line.get_end()]]
    bool_table = [t == np.float64 for t in type_table]
    assert all(bool_table), "Types of start and end points are not np.float64"


def test_type_conversion_in_Arrow3D():
    start, end = [0, 0, 0], [1, 1, 1]
    arrow = Arrow3D(start, end)
    type_table = [type(item) for item in [*arrow.get_start(), *arrow.get_end()]]
    bool_table = [t == np.float64 for t in type_table]
    assert all(bool_table), "Types of start and end points are not np.float64"

    assert np.allclose(arrow.get_start(), start, atol=0.01), (
        "start points of Arrow3D do not match"
    )
    assert np.allclose(arrow.get_end(), end, atol=0.01), (
        "end points of Arrow3D do not match"
    )
