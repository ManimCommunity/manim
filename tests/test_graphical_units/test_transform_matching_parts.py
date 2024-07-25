from __future__ import annotations

from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "transform_matching_parts"


@frames_comparison(last_frame=True)
def test_TransformMatchingLeavesOneObject(scene):
    square = Square()
    circle = Circle().shift(RIGHT)
    scene.add(square)
    scene.play(TransformMatchingShapes(square, circle))
    assert len(scene.mobjects) == 1
    assert isinstance(scene.mobjects[0], Circle)


@frames_comparison(last_frame=False)
def test_TransformMatchingDisplaysCorrect(scene):
    square = Square()
    circle = Circle().shift(RIGHT)
    scene.add(square)
    scene.play(TransformMatchingShapes(square, circle))
    # Wait to make sure object isn't missing in-between animations
    scene.wait(0.5)
    # Shift to make sure object isn't duplicated if moved
    scene.play(circle.animate.shift(DOWN))


@frames_comparison(last_frame=False)
def test_TransformMatchingTex(scene):
    start = MathTex("A", "+", "B", "=", "C")
    end = MathTex("C", "=", "B", "-", "A")

    scene.add(start)
    scene.play(TransformMatchingTex(start, end))


@frames_comparison(last_frame=False)
def test_TransformMatchingTex_FadeTransformMismatches(scene):
    start = MathTex("A", "+", "B", "=", "C")
    end = MathTex("C", "=", "B", "-", "A")

    scene.add(start)
    scene.play(TransformMatchingTex(start, end, fade_transform_mismatches=True))


@frames_comparison(last_frame=False)
def test_TransformMatchingTex_TransformMismatches(scene):
    start = MathTex("A", "+", "B", "=", "C")
    end = MathTex("C", "=", "B", "-", "A")

    scene.add(start)
    scene.play(TransformMatchingTex(start, end, transform_mismatches=True))


@frames_comparison(last_frame=False)
def test_TransformMatchingTex_FadeTransformMismatches_NothingToFade(scene):
    # https://github.com/ManimCommunity/manim/issues/2845
    start = MathTex("A", r"\to", "B")
    end = MathTex("B", r"\to", "A")

    scene.add(start)
    scene.play(TransformMatchingTex(start, end, fade_transform_mismatches=True))
