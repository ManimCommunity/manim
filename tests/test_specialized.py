from manim import *
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

__module_test__ = "specialized"


@frames_comparison(last_frame=False)
def test_broadcast(scene):
    square = Square()
    scene.play(Broadcast(square))
