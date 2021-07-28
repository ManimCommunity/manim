import pytest

from manim import *
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

from ..utils.GraphicalUnitTester import GraphicalUnitTester
from ..utils.testing_utils import get_scenes_to_test

__module_test__ = "mobjects"


@frames_comparison(base_scene=ThreeDScene)
def test_PointCloudDot(scene):
    p = PointCloudDot()
    scene.add(p)
