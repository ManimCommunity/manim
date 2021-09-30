import numpy as np
import pytest

from manim.animation.creation import Uncreate
from manim.mobject.geometry import Dot, Line, Square
from manim.mobject.mobject import override_animate
from manim.mobject.types.vectorized_mobject import VGroup


def test_simple_animate(using_opengl_renderer):
    s = Square()
    scale_factor = 2
    anim = s.animate.scale(scale_factor).build()
    assert anim.mobject.target.width == scale_factor * s.width


def test_chained_animate(using_opengl_renderer):
    s = Square()
    scale_factor = 2
    direction = np.array((1, 1, 0))
    anim = s.animate.scale(scale_factor).shift(direction).build()
    assert (
        anim.mobject.target.width == scale_factor * s.width
        and (anim.mobject.target.get_center() == direction).all()
    )


def test_overridden_animate(using_opengl_renderer):
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


def test_chaining_overridden_animate(using_opengl_renderer):
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


def test_animate_with_args(using_opengl_renderer):
    s = Square()
    scale_factor = 2
    run_time = 2

    anim = s.animate(run_time=run_time).scale(scale_factor).build()
    assert anim.mobject.target.width == scale_factor * s.width
    assert anim.run_time == run_time


def test_chained_animate_with_args(using_opengl_renderer):
    s = Square()
    scale_factor = 2
    direction = np.array((1, 1, 0))
    run_time = 2

    anim = s.animate(run_time=run_time).scale(scale_factor).shift(direction).build()
    assert (
        anim.mobject.target.width == scale_factor * s.width
        and (anim.mobject.target.get_center() == direction).all()
    )
    assert anim.run_time == run_time


def test_animate_with_args_misplaced(using_opengl_renderer):
    s = Square()
    scale_factor = 2
    run_time = 2

    with pytest.raises(ValueError, match="must be passed before"):
        s.animate.scale(scale_factor)(run_time=run_time)

    with pytest.raises(ValueError, match="must be passed before"):
        s.animate(run_time=run_time)(run_time=run_time).scale(scale_factor)
