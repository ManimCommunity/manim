from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from manim.animation.animation import Animation, Wait
from manim.animation.composition import AnimationGroup, Succession
from manim.animation.creation import Create, Write
from manim.animation.fading import FadeIn, FadeOut
from manim.constants import DOWN, UP
from manim.mobject.geometry.arc import Circle
from manim.mobject.geometry.line import Line
from manim.mobject.geometry.polygram import RegularPolygon, Square
from manim.scene.scene import Scene


def test_succession_timing():
    """Test timing of animations in a succession."""
    line = Line()
    animation_1s = FadeIn(line, shift=UP, run_time=1.0)
    animation_4s = FadeOut(line, shift=DOWN, run_time=4.0)
    succession = Succession(animation_1s, animation_4s)
    assert succession.get_run_time() == 5.0
    succession._setup_scene(MagicMock())
    succession.begin()
    assert succession.active_index == 0
    # The first animation takes 20% of the total run time.
    succession.interpolate(0.199)
    assert succession.active_index == 0
    succession.interpolate(0.2)
    assert succession.active_index == 1
    succession.interpolate(0.8)
    assert succession.active_index == 1
    # At 100% and more, no animation must be active anymore.
    succession.interpolate(1.0)
    assert succession.active_index == 2
    assert succession.active_animation is None
    succession.interpolate(1.2)
    assert succession.active_index == 2
    assert succession.active_animation is None


def test_succession_in_succession_timing():
    """Test timing of nested successions."""
    line = Line()
    animation_1s = FadeIn(line, shift=UP, run_time=1.0)
    animation_4s = FadeOut(line, shift=DOWN, run_time=4.0)
    nested_succession = Succession(animation_1s, animation_4s)
    succession = Succession(
        FadeIn(line, shift=UP, run_time=4.0),
        nested_succession,
        FadeIn(line, shift=UP, run_time=1.0),
    )
    assert nested_succession.get_run_time() == 5.0
    assert succession.get_run_time() == 10.0
    succession._setup_scene(MagicMock())
    succession.begin()
    succession.interpolate(0.1)
    assert succession.active_index == 0
    # The nested succession must not be active yet, and as a result hasn't set active_animation yet.
    assert not hasattr(nested_succession, "active_animation")
    succession.interpolate(0.39)
    assert succession.active_index == 0
    assert not hasattr(nested_succession, "active_animation")
    # The nested succession starts at 40% of total run time
    succession.interpolate(0.4)
    assert succession.active_index == 1
    assert nested_succession.active_index == 0
    # The nested succession second animation starts at 50% of total run time.
    succession.interpolate(0.49)
    assert succession.active_index == 1
    assert nested_succession.active_index == 0
    succession.interpolate(0.5)
    assert succession.active_index == 1
    assert nested_succession.active_index == 1
    # The last animation starts at 90% of total run time. The nested succession must be finished at that time.
    succession.interpolate(0.89)
    assert succession.active_index == 1
    assert nested_succession.active_index == 1
    succession.interpolate(0.9)
    assert succession.active_index == 2
    assert nested_succession.active_index == 2
    assert nested_succession.active_animation is None
    # After 100%, nothing must be playing anymore.
    succession.interpolate(1.0)
    assert succession.active_index == 3
    assert succession.active_animation is None
    assert nested_succession.active_index == 2
    assert nested_succession.active_animation is None


def test_animationbuilder_in_group():
    sqr = Square()
    circ = Circle()
    animation_group = AnimationGroup(sqr.animate.shift(DOWN).scale(2), FadeIn(circ))
    assert all(isinstance(anim, Animation) for anim in animation_group.animations)
    succession = Succession(sqr.animate.shift(DOWN).scale(2), FadeIn(circ))
    assert all(isinstance(anim, Animation) for anim in succession.animations)


def test_animationgroup_with_wait():
    sqr = Square()
    sqr_anim = FadeIn(sqr)
    wait = Wait()
    animation_group = AnimationGroup(wait, sqr_anim, lag_ratio=1)

    animation_group.begin()
    timings = animation_group.anims_with_timings

    assert timings == [(wait, 0.0, 1.0), (sqr_anim, 1.0, 2.0)]


@pytest.mark.parametrize(
    "animation_remover, animation_group_remover",
    [(False, True), (True, False)],
)
def test_animationgroup_is_passing_remover_to_animations(
    animation_remover, animation_group_remover
):
    scene = Scene()
    sqr_animation = Create(Square(), remover=animation_remover)
    circ_animation = Write(Circle(), remover=animation_remover)
    animation_group = AnimationGroup(
        sqr_animation, circ_animation, remover=animation_group_remover
    )

    scene.play(animation_group)
    scene.wait(0.1)

    assert sqr_animation.remover
    assert circ_animation.remover


def test_animationgroup_is_passing_remover_to_nested_animationgroups():
    scene = Scene()
    sqr_animation = Create(Square())
    circ_animation = Write(Circle(), remover=True)
    polygon_animation = Create(RegularPolygon(5))
    animation_group = AnimationGroup(
        AnimationGroup(sqr_animation, polygon_animation),
        circ_animation,
        remover=True,
    )

    scene.play(animation_group)
    scene.wait(0.1)

    assert sqr_animation.remover
    assert circ_animation.remover
    assert polygon_animation.remover
