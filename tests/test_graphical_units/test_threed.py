import pytest

from manim import *

from ..utils.GraphicalUnitTester import GraphicalUnitTester
from ..utils.testing_utils import get_scenes_to_test


class CubeTest(ThreeDScene):
    def construct(self):
        self.add(Cube())


class SphereTest(ThreeDScene):
    def construct(self):
        self.add(Sphere())


class Dot3DTest(ThreeDScene):
    def construct(self):
        self.add(Dot3D())


class ConeTest(ThreeDScene):
    def construct(self):
        self.add(Cone())


class CylinderTest(ThreeDScene):
    def construct(self):
        self.add(Cylinder())


class Line3DTest(ThreeDScene):
    def construct(self):
        self.add(Line3D())


class Arrow3DTest(ThreeDScene):
    def construct(self):
        self.add(Arrow3D())


class TorusTest(ThreeDScene):
    def construct(self):
        self.add(Torus())


class AxesTest(ThreeDScene):
    def construct(self):
        self.add(ThreeDAxes(axis_config={"exclude_origin_tick": False}))


class CameraMoveTest(ThreeDScene):
    def construct(self):
        cube = Cube()
        self.add(cube)
        self.move_camera(phi=PI / 4, theta=PI / 4, frame_center=[0, 0, -1])


class AmbientCameraMoveTest(ThreeDScene):
    def construct(self):
        cube = Cube()
        self.begin_ambient_camera_rotation(rate=0.5)
        self.add(cube)
        self.wait()


# TODO: bring test back after introducing testing tolerance
#  to account for OS-specific differences in numerics.

# class FixedInFrameMObjectTest(ThreeDScene):
#     def construct(self):
#         axes = ThreeDAxes()
#         self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
#         circ = Circle()
#         self.add_fixed_in_frame_mobjects(circ)
#         circ.to_corner(UL)
#         self.add(axes)


class MovingVerticesTest(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        vertices = [1, 2, 3, 4]
        edges = [(1, 2), (2, 3), (3, 4), (1, 3), (1, 4)]
        g = Graph(vertices, edges)
        self.add(g)
        self.play(
            g[1].animate.move_to([1, 1, 1]),
            g[2].animate.move_to([-1, 1, 2]),
            g[3].animate.move_to([1, -1, -1]),
            g[4].animate.move_to([-1, -1, 0]),
        )
        self.wait()


class SurfaceColorscaleTest(ThreeDScene):
    def construct(self):
        resolution_fa = 50
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        axes = ThreeDAxes(x_range=(-3, 3, 1), y_range=(-3, 3, 1), z_range=(-4, 4, 1))

        def param_trig(u, v):
            x = u
            y = v
            z = y ** 2 / 2 - x ** 2 / 2
            return z

        trig_plane = ParametricSurface(
            lambda x, y: axes.c2p(x, y, param_trig(x, y)),
            resolution=(resolution_fa, resolution_fa),
            v_min=-3,
            v_max=+3,
            u_min=-3,
            u_max=+3,
        )

        trig_plane.set_fill_by_value(
            axes=axes, colors=[BLUE, GREEN, YELLOW, ORANGE, RED]
        )
        self.add(axes, trig_plane)


MODULE_NAME = "threed"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
