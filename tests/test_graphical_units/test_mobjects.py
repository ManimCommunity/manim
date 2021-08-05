from manim import *
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

__module_test__ = "mobjects"


@frames_comparison(base_scene=ThreeDScene)
def test_PointCloudDot(scene):
    p = PointCloudDot()
    scene.add(p)
