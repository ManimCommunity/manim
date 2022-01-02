from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "modifier_methods"


@frames_comparison
def test_Gradient(scene):
    c = Circle(fill_opacity=1).set_color(color=[YELLOW, GREEN])
    scene.add(c)


@frames_comparison
def test_GradientRotation(scene):
    c = Circle(fill_opacity=1).set_color(color=[YELLOW, GREEN]).rotate(PI)
    scene.add(c)
