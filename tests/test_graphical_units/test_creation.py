from manim import *
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

__module_test__ = "creation"


@frames_comparison(last_frame=False)
def test_create(scene):
    square = Square()
    scene.play(Create(square))


@frames_comparison(last_frame=False)
def test_uncreate(scene):
    square = Square()
    scene.add(square)
    scene.play(Uncreate(square))


@frames_comparison(last_frame=False)
def test_DrawBorderThenFill(scene):
    square = Square(fill_opacity=1)
    scene.play(DrawBorderThenFill(square))


# NOTE : Here should be the Write Test. But for some reasons it appears that this function is untestable (see issue #157)
@frames_comparison(last_frame=False)
def test_FadeOut(scene):
    square = Square()
    scene.add(square)
    scene.play(FadeOut(square))


@frames_comparison(last_frame=False)
def test_FadeIn(scene):
    square = Square()
    scene.play(FadeIn(square))


@frames_comparison(last_frame=False)
def test_GrowFromPoint(scene):
    square = Square()
    scene.play(GrowFromPoint(square, np.array((1, 1, 0))))


@frames_comparison(last_frame=False)
def test_GrowFromCenter(scene):
    square = Square()
    scene.play(GrowFromCenter(square))


@frames_comparison(last_frame=False)
def test_GrowFromEdge(scene):
    square = Square()
    scene.play(GrowFromEdge(square, DOWN))


@frames_comparison(last_frame=False)
def test_SpinInFromNothing(scene):
    square = Square()
    scene.play(SpinInFromNothing(square))


@frames_comparison(last_frame=False)
def test_ShrinkToCenter(scene):
    square = Square()
    scene.play(ShrinkToCenter(square))
