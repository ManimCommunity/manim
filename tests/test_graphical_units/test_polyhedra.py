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
