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


MODULE_NAME = "threed"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
