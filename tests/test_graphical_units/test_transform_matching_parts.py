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
    assert len(scene.mobjects) == 1 and isinstance(scene.mobjects[0], Circle)


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
