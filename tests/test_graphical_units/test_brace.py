from manim import *
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

__module_test__ = "brace"


@frames_comparison
def test_brace_sharpness(scene):
    line = Line(LEFT * 3, RIGHT * 3).shift(UP * 4)
    for sharpness in [0, 0.25, 0.5, 0.75, 1, 2, 3, 5]:
        scene.add(Brace(line, sharpness=sharpness))
        line.shift(DOWN)
        scene.wait()


@frames_comparison
def test_braceTip(scene):
    line = Line().shift(LEFT * 3).rotate(PI / 2)
    steps = 8
    for _i in range(steps):
        brace = Brace(line, direction=line.copy().rotate(PI / 2).get_unit_vector())
        dot = Dot()
        brace.put_at_tip(dot)
        line.rotate_about_origin(TAU / steps)
        scene.add(brace, dot)
        scene.wait()


@frames_comparison
def test_arcBrace(scene):
    scene.play(Animation(ArcBrace()))
