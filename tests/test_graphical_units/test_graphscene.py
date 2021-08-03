from manim import *
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

__module_test__ = "graphscene"


@frames_comparison(base_scene=GraphScene)
def test_PlotFunctions(scene):
    scene.x_min = -10
    scene.x_max = 10.3
    scene.y_min = -1.5
    scene.y_max = 1.5
    constants.TEX_TEMPLATE = TexTemplate()
    scene.setup_axes()
    f = scene.get_graph(lambda x: x ** 2)
    scene.add(f)
