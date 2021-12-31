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
    scene.add(Cone())


@frames_comparison(base_scene=ThreeDScene)
def test_Cylinder(scene):
    scene.add(Cylinder())


@frames_comparison(base_scene=ThreeDScene)
def test_Line3D(scene):
    line = Line3D()
    perp_line = Line3D.perpendicular_to(line, UP)
    parallel_line = Line3D.parallel_to(line, UP)
    scene.add(line, perp_line, parallel_line)


@frames_comparison(base_scene=ThreeDScene)
def test_Arrow3D(scene):
    scene.add(Arrow3D())


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


# TODO: bring test back after introducing testing tolerance
#  to account for OS-specific differences in numerics.

# class FixedInFrameMObjectTest(ThreeDScene):
#     def construct(scene):
#         axes = ThreeDAxes()
#         scene.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
#         circ = Circle()
#         scene.add_fixed_in_frame_mobjects(circ)
#         circ.to_corner(UL)
#         scene.add(axes)


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
    resolution_fa = 50
    scene.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)
    axes = ThreeDAxes(x_range=(-3, 3, 1), y_range=(-3, 3, 1), z_range=(-4, 4, 1))

    def param_trig(u, v):
        x = u
        y = v
        z = y ** 2 / 2 - x ** 2 / 2
        return z

    trig_plane = Surface(
        lambda x, y: axes.c2p(x, y, param_trig(x, y)),
        resolution=(resolution_fa, resolution_fa),
        v_range=[-3, 3],
        u_range=[-3, 3],
    )
    trig_plane.set_fill_by_value(axes=axes, colors=[BLUE, GREEN, YELLOW, ORANGE, RED])
    scene.add(axes, trig_plane)


@frames_comparison(base_scene=ThreeDScene)
def test_Y_Direction(scene):
    resolution_fa = 42
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
        axes=axes, colors=[(RED, -0.4), (YELLOW, 0), (GREEN, 0.4)], axis=1
    )
    scene.add(axes, surface_plane)
