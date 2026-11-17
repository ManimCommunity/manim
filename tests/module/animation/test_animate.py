from __future__ import annotations

import numpy as np
import pytest

from manim.animation.animation import Animation
from manim.animation.creation import Uncreate
from manim.constants import RIGHT
from manim.mobject.geometry.arc import Circle, Dot
from manim.mobject.geometry.line import Line
from manim.mobject.geometry.polygram import Square
from manim.mobject.mobject import override_animate
from manim.mobject.text.numbers import Integer
from manim.mobject.types.vectorized_mobject import VGroup


def test_simple_animate():
    s = Square()
    scale_factor = 2
    anim = s.animate.scale(scale_factor).build()
    assert anim.mobject.target.width == scale_factor * s.width


def test_chained_animate():
    s = Square()
    scale_factor = 2
    direction = np.array((1, 1, 0))
    anim = s.animate.scale(scale_factor).shift(direction).build()
    assert anim.mobject.target.width == scale_factor * s.width
    assert (anim.mobject.target.get_center() == direction).all()


def test_overridden_animate():
    class DotsWithLine(VGroup):
        def __init__(self):
            super().__init__()
            self.left_dot = Dot().shift((-1, 0, 0))
            self.right_dot = Dot().shift((1, 0, 0))
            self.line = Line(self.left_dot, self.right_dot)
            self.add(self.left_dot, self.right_dot, self.line)

        def remove_line(self):
            self.remove(self.line)

        @override_animate(remove_line)
        def _remove_line_animation(self, anim_args=None):
            if anim_args is None:
                anim_args = {}
            self.remove_line()
            return Uncreate(self.line, **anim_args)

    dots_with_line = DotsWithLine()
    anim = dots_with_line.animate.remove_line().build()
    assert len(dots_with_line.submobjects) == 2
    assert type(anim) is Uncreate


def test_chaining_overridden_animate():
    class DotsWithLine(VGroup):
        def __init__(self):
            super().__init__()
            self.left_dot = Dot().shift((-1, 0, 0))
            self.right_dot = Dot().shift((1, 0, 0))
            self.line = Line(self.left_dot, self.right_dot)
            self.add(self.left_dot, self.right_dot, self.line)

        def remove_line(self):
            self.remove(self.line)

        @override_animate(remove_line)
        def _remove_line_animation(self, anim_args=None):
            if anim_args is None:
                anim_args = {}
            self.remove_line()
            return Uncreate(self.line, **anim_args)

    with pytest.raises(
        NotImplementedError,
        match="not supported for overridden animations",
    ):
        DotsWithLine().animate.shift((1, 0, 0)).remove_line()

    with pytest.raises(
        NotImplementedError,
        match="not supported for overridden animations",
    ):
        DotsWithLine().animate.remove_line().shift((1, 0, 0))


def test_animate_with_args():
    s = Square()
    scale_factor = 2
    run_time = 2

    anim = s.animate(run_time=run_time).scale(scale_factor).build()
    assert anim.mobject.target.width == scale_factor * s.width
    assert anim.run_time == run_time


def test_chained_animate_with_args():
    s = Square()
    scale_factor = 2
    direction = np.array((1, 1, 0))
    run_time = 2

    anim = s.animate(run_time=run_time).scale(scale_factor).shift(direction).build()
    assert anim.mobject.target.width == scale_factor * s.width
    assert (anim.mobject.target.get_center() == direction).all()
    assert anim.run_time == run_time


def test_animate_with_args_misplaced():
    s = Square()
    scale_factor = 2
    run_time = 2

    with pytest.raises(ValueError, match="must be passed before"):
        s.animate.scale(scale_factor)(run_time=run_time)

    with pytest.raises(ValueError, match="must be passed before"):
        s.animate(run_time=run_time)(run_time=run_time).scale(scale_factor)


def _run_without_scene(anim: Animation) -> None:
    anim.begin()
    anim.interpolate(0.5)
    anim.finish()


@pytest.mark.parametrize(("start", "end"), [(10, 0), (100, 25), (1000, 5)])
def test_animate_set_value_with_fewer_digits(start, end):
    # Regression test for https://github.com/ManimCommunity/manim/issues/5141
    number = Integer(start)
    anim = number.animate.set_value(end).build()
    _run_without_scene(anim)

    assert number.get_value() == end
    # Transform may leave invisible copies used for alignment; only the
    # visible digits have to match the target.
    visible = [
        mob for mob in number.family_members_with_points() if mob.get_fill_opacity() > 0
    ]
    expected = anim.target_mobject.family_members_with_points()
    assert len(visible) == len(expected)
    for mob, expected_mob in zip(visible, expected, strict=True):
        np.testing.assert_allclose(mob.get_center(), expected_mob.get_center())
        np.testing.assert_allclose(mob.width, expected_mob.width)
        np.testing.assert_allclose(mob.height, expected_mob.height)


def test_animate_add_submobject():
    group = VGroup(Square())
    circle = Circle().shift(2 * RIGHT)
    anim = group.animate.add(circle).build()
    _run_without_scene(anim)

    assert circle in group.submobjects
