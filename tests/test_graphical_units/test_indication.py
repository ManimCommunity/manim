from manim import *
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

__module_test__ = "indication"


@frames_comparison(last_frame=False)
def test_FocusOn(scene):
    square = Square()
    scene.add(square)
    scene.play(FocusOn(square))


@frames_comparison(last_frame=False)
def test_Indicate(scene):
    square = Square()
    scene.add(square)
    scene.play(Indicate(square))


@frames_comparison(last_frame=False)
def test_Flash(scene):
    square = Square()
    scene.add(square)
    scene.play(Flash(ORIGIN))


@frames_comparison(last_frame=False)
def test_Circumscribe(scene):
    square = Square()
    scene.add(square)
    scene.play(Circumscribe(square))
    scene.wait()


@frames_comparison(last_frame=False)
def test_ShowPassingFlash(scene):
    square = Square()
    scene.add(square)
    scene.play(ShowPassingFlash(square.copy()))


@frames_comparison(last_frame=False)
def test_ShowCreationThenFadeOut(scene):
    square = Square()
    scene.add(square)
    scene.play(ShowCreationThenFadeOut(square.copy()))


@frames_comparison(last_frame=False)
def test_ApplyWave(scene):
    square = Square()
    scene.add(square)
    scene.play(ApplyWave(square))


@frames_comparison(last_frame=False)
def test_Wiggle(scene):
    square = Square()
    scene.add(square)
    scene.play(Wiggle(square))
