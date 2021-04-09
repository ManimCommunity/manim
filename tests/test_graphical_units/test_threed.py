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
        self.add(ThreeDAxes())


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


class FixedInFrameMObjectTest(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        circ = Circle()
        self.add_fixed_in_frame_mobjects(circ)
        circ.to_corner(UL)
        self.add(axes)


MODULE_NAME = "threed"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
