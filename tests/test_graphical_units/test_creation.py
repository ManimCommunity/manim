from __future__ import annotations

from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

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
def test_uncreate_rate_func(scene):
    square = Square()
    scene.add(square)
    scene.play(Uncreate(square), rate_func=linear)


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


@frames_comparison(last_frame=False)
def test_bring_to_back_introducer(scene):
    a = Square(color=RED, fill_opacity=1)
    b = Square(color=BLUE, fill_opacity=1).shift(RIGHT)
    scene.add(a)
    scene.bring_to_back(b)
    scene.play(FadeIn(b))
    scene.wait()


@frames_comparison(last_frame=False)
def test_z_index_introducer(scene):
    a = Circle().set_fill(color=RED, opacity=1.0)
    scene.add(a)
    b = Circle(arc_center=(0.5, 0.5, 0.0), color=GREEN, fill_opacity=1)
    b.set_z_index(-1)
    scene.play(Create(b))
    scene.wait()


@frames_comparison(last_frame=False)
def test_SpiralIn(scene):
    circle = Circle().shift(LEFT)
    square = Square().shift(UP)
    shapes = VGroup(circle, square)
    scene.play(SpiralIn(shapes))
