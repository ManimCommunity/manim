"""Utility functions for continuous animation of mobjects."""

from __future__ import annotations

__all__ = [
    "assert_is_mobject_method",
    "always",
    "f_always",
    "always_redraw",
    "always_shift",
    "always_rotate",
    "turn_animation_into_updater",
    "cycle_animation",
]


import inspect
from typing import TYPE_CHECKING, Callable

import numpy as np

from manim.constants import DEGREES, RIGHT
from manim.mobject.mobject import Mobject
from manim.opengl import OpenGLMobject
from manim.utils.space_ops import normalize

if TYPE_CHECKING:
    from manim.animation.animation import Animation


def assert_is_mobject_method(method: Callable) -> None:
    assert inspect.ismethod(method)
    mobject = method.__self__
    assert isinstance(mobject, (Mobject, OpenGLMobject))


def always(method: Callable, *args, **kwargs) -> Mobject:
    assert_is_mobject_method(method)
    mobject = method.__self__
    func = method.__func__
    mobject.add_updater(lambda m: func(m, *args, **kwargs))
    return mobject


def f_always(method: Callable[[Mobject], None], *arg_generators, **kwargs) -> Mobject:
    """
    More functional version of always, where instead
    of taking in args, it takes in functions which output
    the relevant arguments.
    """
    assert_is_mobject_method(method)
    mobject = method.__self__
    func = method.__func__

    def updater(mob):
        args = [arg_generator() for arg_generator in arg_generators]
        func(mob, *args, **kwargs)

    mobject.add_updater(updater)
    return mobject


def always_redraw(func: Callable[[], Mobject]) -> Mobject:
    """Redraw the mobject constructed by a function every frame.

    This function returns a mobject with an attached updater that
    continuously regenerates the mobject according to the
    specified function.

    Parameters
    ----------
    func
        A function without (required) input arguments that returns
        a mobject.

    Examples
    --------

    .. manim:: TangentAnimation

        class TangentAnimation(Scene):
            def construct(self):
                ax = Axes()
                sine = ax.plot(np.sin, color=RED)
                alpha = ValueTracker(0)
                point = always_redraw(
                    lambda: Dot(
                        sine.point_from_proportion(alpha.get_value()),
                        color=BLUE
                    )
                )
                tangent = always_redraw(
                    lambda: TangentLine(
                        sine,
                        alpha=alpha.get_value(),
                        color=YELLOW,
                        length=4
                    )
                )
                self.add(ax, sine, point, tangent)
                self.play(alpha.animate.set_value(1), rate_func=linear, run_time=2)
    """
    mob = func()
    mob.add_updater(lambda _: mob.become(func()))
    return mob


def always_shift(
    mobject: Mobject, direction: np.ndarray[np.float64] = RIGHT, rate: float = 0.1
) -> Mobject:
    """A mobject which is continuously shifted along some direction
    at a certain rate.

    Parameters
    ----------
    mobject
        The mobject to shift.
    direction
        The direction to shift. The vector is normalized, the specified magnitude
        is not relevant.
    rate
        Length in Manim units which the mobject travels in one
        second along the specified direction.

    Examples
    --------

    .. manim:: ShiftingSquare

        class ShiftingSquare(Scene):
            def construct(self):
                sq = Square().set_fill(opacity=1)
                tri = Triangle()
                VGroup(sq, tri).arrange(LEFT)

                # construct a square which is continuously
                # shifted to the right
                always_shift(sq, RIGHT, rate=5)

                self.add(sq)
                self.play(tri.animate.set_fill(opacity=1))
    """
    mobject.add_updater(lambda m, dt: m.shift(dt * rate * normalize(direction)))
    return mobject


def always_rotate(mobject: Mobject, rate: float = 20 * DEGREES, **kwargs) -> Mobject:
    """A mobject which is continuously rotated at a certain rate.

    Parameters
    ----------
    mobject
        The mobject to be rotated.
    rate
        The angle which the mobject is rotated by
        over one second.
    kwags
        Further arguments to be passed to :meth:`.Mobject.rotate`.

    Examples
    --------

    .. manim:: SpinningTriangle

        class SpinningTriangle(Scene):
            def construct(self):
                tri = Triangle().set_fill(opacity=1).set_z_index(2)
                sq = Square().to_edge(LEFT)

                # will keep spinning while there is an animation going on
                always_rotate(tri, rate=2*PI, about_point=ORIGIN)

                self.add(tri, sq)
                self.play(sq.animate.to_edge(RIGHT), rate_func=linear, run_time=1)
    """
    mobject.add_updater(lambda m, dt: m.rotate(dt * rate, **kwargs))
    return mobject


def turn_animation_into_updater(
    animation: Animation, cycle: bool = False, delay: float = 0, **kwargs
) -> Mobject:
    """
    Add an updater to the animation's mobject which applies
    the interpolation and update functions of the animation

    If cycle is True, this repeats over and over.  Otherwise,
    the updater will be popped upon completion

    The ``delay`` parameter is the delay (in seconds) before the animation starts..

    Examples
    --------

    .. manim:: WelcomeToManim

        class WelcomeToManim(Scene):
            def construct(self):
                words = Text("Welcome to")
                banner = ManimBanner().scale(0.5)
                VGroup(words, banner).arrange(DOWN)

                turn_animation_into_updater(Write(words, run_time=0.9))
                self.add(words)
                self.wait(0.5)
                self.play(banner.expand(), run_time=0.5)
    """
    mobject = animation.mobject
    animation.suspend_mobject_updating = False
    animation.begin()
    animation.total_time = -delay

    def update(m: Mobject, dt: float):
        if animation.total_time >= 0:
            run_time = animation.get_run_time()
            time_ratio = animation.total_time / run_time
            if cycle:
                alpha = time_ratio % 1
            else:
                alpha = np.clip(time_ratio, 0, 1)
                if alpha >= 1:
                    animation.finish()
                    m.remove_updater(update)
                    return
            animation.interpolate(alpha)
            animation.update_mobjects(dt)
        animation.total_time += dt

    mobject.add_updater(update)
    return mobject


def cycle_animation(animation: Animation, **kwargs) -> Mobject:
    return turn_animation_into_updater(animation, cycle=True, **kwargs)
