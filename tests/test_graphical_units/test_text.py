from manim import *
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

__module_test__ = "text"


@frames_comparison
def test_Text2Color(scene):
    scene.add(Text("this is  a text  with spaces!", t2c={"spaces": RED}))
