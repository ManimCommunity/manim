from __future__ import annotations

from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

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
def test_ApplyWave(scene):
    square = Square()
    scene.add(square)
    scene.play(ApplyWave(square))


@frames_comparison(last_frame=False)
def test_Wiggle(scene):
    square = Square()
    scene.add(square)
    scene.play(Wiggle(square))


def test_Wiggle_custom_about_points():
    square = Square()
    wiggle = Wiggle(
        square,
        scale_about_point=[1.0, 2.0, 3.0],
        rotate_about_point=[4.0, 5.0, 6.0],
    )
    assert np.all(wiggle.get_scale_about_point() == [1.0, 2.0, 3.0])
    assert np.all(wiggle.get_rotate_about_point() == [4.0, 5.0, 6.0])
