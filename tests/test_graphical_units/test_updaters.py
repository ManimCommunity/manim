from manim import *
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

__module_test__ = "updaters"


@frames_comparison(last_frame=False)
def test_Updater(scene):
    dot = Dot()
    square = Square()
    scene.add(dot, square)
    square.add_updater(lambda m: m.next_to(dot, RIGHT, buff=SMALL_BUFF))
    scene.add(square)
    scene.play(dot.animate.shift(UP * 2))
    square.clear_updaters()


@frames_comparison
def test_ValueTracker(scene):
    theta = ValueTracker(PI / 2)
    line = Line(ORIGIN, RIGHT)
    line.rotate(theta.get_value(), about_point=ORIGIN)
    scene.add(line)


@frames_comparison(last_frame=False)
def test_UpdateSceneDuringAnimation(scene):
    def f(mob):
        scene.add(Square())

    s = Circle().add_updater(f)
    scene.play(Create(s))


@frames_comparison(last_frame=False)
def test_LastFrameWhenCleared(scene):
    dot = Dot()
    square = Square()
    square.add_updater(lambda m: m.move_to(dot, UL))
    scene.add(square)
    scene.play(dot.animate.shift(UP * 2), rate_func=linear)
    square.clear_updaters()
    scene.wait()
