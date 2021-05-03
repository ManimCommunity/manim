import pytest

from manim import *

from ..utils.GraphicalUnitTester import GraphicalUnitTester
from ..utils.testing_utils import get_scenes_to_test


class TetrahedronTest(ThreeDScene):
    def construct(self):
        self.add(Tetrahedron())


class OctahedronTest(ThreeDScene):
    def construct(self):
        self.add(Octahedron())


class IcosahedronTest(ThreeDScene):
    def construct(self):
        self.add(Icosahedron())


class DodecahedronTest(ThreeDScene):
    def construct(self):
        self.add(Dodecahedron())


MODULE_NAME = "polyhedra"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
