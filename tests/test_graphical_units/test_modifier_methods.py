from manim import *
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

from ..utils.GraphicalUnitTester import GraphicalUnitTester
from ..utils.testing_utils import get_scenes_to_test

__module_test__ = "modifier_methods"


@frames_comparison
def test_Gradient(scene):
    c = Circle(fill_opacity=1).set_color(color=[YELLOW, GREEN])
    scene.add(c)


@frames_comparison
def test_GradientRotation(scene):
    c = Circle(fill_opacity=1).set_color(color=[YELLOW, GREEN]).rotate(PI)
    scene.add(c)
