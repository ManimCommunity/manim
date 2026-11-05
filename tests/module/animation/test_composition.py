from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from manim.animation.animation import Animation, Wait
from manim.animation.composition import AnimationGroup, LaggedStartMap, Succession
from manim.animation.creation import Create, Write
from manim.animation.fading import FadeIn, FadeOut
from manim.animation.transform import Transform
from manim.constants import DOWN, RIGHT, UP
from manim.mobject.geometry.arc import Circle
from manim.mobject.geometry.line import Line
from manim.mobject.geometry.polygram import RegularPolygon, Square
from manim.mobject.types.vectorized_mobject import VGroup
from manim.scene.scene import Scene
from manim.utils.rate_functions import linear, there_and_back


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


def test_timescaled_succession():
    s1, s2, s3 = Square(), Square(), Square()
    anim = Succession(
        FadeIn(s1, run_time=2),
        FadeIn(s2),
        FadeIn(s3),
    )
    anim.scene = MagicMock()
    anim.run_time = 42
    anim.begin()
    anim.interpolate(0.2)
    assert anim.active_index == 0
    anim.interpolate(0.4)
    assert anim.active_index == 0
    anim.interpolate(0.6)
    assert anim.active_index == 1
    anim.interpolate(0.8)
    assert anim.active_index == 2


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

    assert timings.tolist() == [(wait, 0.0, 1.0), (sqr_anim, 1.0, 2.0)]


@pytest.mark.parametrize(
    ("animation_remover", "animation_group_remover"),
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


def test_animationgroup_calls_finish():
    class MyAnimation(Animation):
        def __init__(self, mobject):
            super().__init__(mobject)
            self.finished = False

        def finish(self):
            self.finished = True

    scene = Scene()
    sqr_animation = MyAnimation(Square())
    circ_animation = MyAnimation(Circle())
    animation_group = AnimationGroup(sqr_animation, circ_animation)
    scene.play(animation_group)
    assert sqr_animation.finished
    assert circ_animation.finished


def test_laggedstartmap_only_passes_kwargs_to_subanimations():
    mobject = VGroup(Square(), Circle())
    animation = LaggedStartMap(
        FadeIn,
        mobject,
        rate_func=there_and_back,
        lag_ratio=0.3,
    )

    assert animation.rate_func is linear
    assert animation.lag_ratio == 0.3
    assert all(
        subanimation.rate_func is there_and_back
        for subanimation in animation.animations
    )


def test_laggedstartmap_default_arg_creator_keeps_submobject_whole():
    inner1 = VGroup(Square(), Square())
    inner2 = VGroup(Circle(), Circle())
    outer = VGroup(inner1, inner2)

    received_args = []

    class RecordingAnimation(Animation):
        def __init__(self, *args, **kwargs):
            received_args.append(args)
            super().__init__(args[0], **kwargs)

    LaggedStartMap(RecordingAnimation, outer)

    assert all(len(args) == 1 for args in received_args)
    assert [args[0] for args in received_args] == [inner1, inner2]


def test_empty_animation_group_fails():
    with pytest.raises(ValueError, match="Please add at least one subanimation."):
        AnimationGroup().begin()


def test_empty_succession_fails():
    with pytest.raises(ValueError, match="Please add at least one subanimation."):
        Succession().begin()


def _move_right_then_up(square):
    return (
        Transform(square, square.copy().shift(2 * RIGHT)),
        Transform(square, square.copy().shift(2 * RIGHT + 2 * UP)),
    )


@pytest.mark.parametrize("nested", [False, True])
def test_animationgroup_begins_subanimations_at_their_start_time(nested):
    scene = Scene()
    square = Square()
    scene.add(square)
    group = AnimationGroup(*_move_right_then_up(square), lag_ratio=1)
    if nested:
        group = AnimationGroup(group)
    group._setup_scene(scene)
    group.begin()

    # The second Transform must start from where the first one ended.
    group.interpolate(0.5)
    np.testing.assert_allclose(square.get_center(), 2 * RIGHT)
    group.interpolate(0.75)
    np.testing.assert_allclose(square.get_center(), 2 * RIGHT + UP)


def test_animationgroup_sets_up_subanimations_at_their_start_time():
    scene = Scene()
    circle = Circle()
    group = AnimationGroup(Wait(1), FadeIn(circle), lag_ratio=1)
    group._setup_scene(scene)
    group.begin()

    group.interpolate(0.25)
    assert circle not in scene.get_mobject_family_members()
    group.interpolate(0.75)
    assert circle in scene.get_mobject_family_members()


@pytest.mark.parametrize(
    "make_group",
    [lambda *anims: AnimationGroup(*anims, lag_ratio=1), Succession],
)
def test_subanimations_are_cleaned_up_when_they_end(make_group):
    scene = Scene()
    square = Square()
    scene.add(square)
    group = make_group(FadeOut(square), FadeIn(square))
    group._setup_scene(scene)
    group.begin()

    # FadeOut has been cleaned up, so FadeIn fades the square in again.
    group.interpolate(0.75)
    assert square.get_stroke_opacity() == pytest.approx(0.5)

    group.finish()
    group.clean_up_from_scene(scene)
    assert square in scene.get_mobject_family_members()
    assert square.get_stroke_opacity() == pytest.approx(1)


def test_animationgroup_subanimations_can_rewind():
    scene = Scene()
    squares = [Square(), Square()]
    scene.add(*squares)
    group = AnimationGroup(
        *(square.animate.shift(UP) for square in squares),
        lag_ratio=0.5,
        rate_func=there_and_back,
    )
    group._setup_scene(scene)
    group.begin()

    group.interpolate(0.5)
    assert [square.get_y() for square in squares] == pytest.approx([1, 1])
    group.interpolate(1)
    assert [square.get_y() for square in squares] == pytest.approx([0, 0])
