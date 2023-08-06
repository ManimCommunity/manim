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
from collections.abc import Callable

import numpy as np

from manim.constants import DEGREES, RIGHT
from manim.mobject.mobject import Mobject
from manim.opengl import OpenGLMobject


def assert_is_mobject_method(method):
    assert inspect.ismethod(method)
    mobject = method.__self__
    assert isinstance(mobject, (Mobject, OpenGLMobject))


def always(method, *args, **kwargs):
    assert_is_mobject_method(method)
    mobject = method.__self__
    func = method.__func__
    mobject.add_updater(lambda m: func(m, *args, **kwargs))
    return mobject


def f_always(method, *arg_generators, **kwargs):
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
                            color=BLUE)
                    )
                tangent = always_redraw(
                    lambda: TangentLine(
                        sine,
                        alpha=alpha.get_value(),
                        color=YELLOW,
                        length=4)
                )
                self.add(ax, sine, point, tangent)
                self.play(alpha.animate.set_value(1), rate_func=linear, run_time=2)
    """
    mob = func()
    mob.add_updater(lambda _: mob.become(func()))
    return mob


def always_shift(mobject, direction=RIGHT, rate=0.1):
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    mobject.add_updater(lambda m, dt: m.shift(dt * rate * normalize(direction)))
    return mobject


def always_rotate(mobject, rate=20 * DEGREES, **kwargs):
    mobject.add_updater(lambda m, dt: m.rotate(dt * rate, **kwargs))
    return mobject


def turn_animation_into_updater(animation, cycle=False, **kwargs):
    """
    Add an updater to the animation's mobject which applies
    the interpolation and update functions of the animation

    If cycle is True, this repeats over and over.  Otherwise,
    the updater will be popped upon completion
    """
    mobject = animation.mobject
    animation.suspend_mobject_updating = False
    animation.begin()
    animation.total_time = 0

    def update(m, dt):
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


def cycle_animation(animation, **kwargs):
    return turn_animation_into_updater(animation, cycle=True, **kwargs)
