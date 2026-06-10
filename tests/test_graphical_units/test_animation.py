from manim import *
from manim.animation.animation import DEFAULT_ANIMATION_RUN_TIME

__module_test__ = "animation"


def test_animation_set_default():
    s = Square()
    Rotate.set_default(run_time=100)
    anim = Rotate(s)
    assert anim.run_time == 100
    anim = Rotate(s, run_time=5)
    assert anim.run_time == 5
    Rotate.set_default()
    anim = Rotate(s)
    assert anim.run_time == DEFAULT_ANIMATION_RUN_TIME
