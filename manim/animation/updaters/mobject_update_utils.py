"""Utility functions for continuous animation of mobjects."""

from __future__ import annotations

__all__ = [
    "assert_is_mobject_method",
    "always_redraw",
    "turn_animation_into_updater",
    "cycle_animation",
]


import inspect
from typing import TYPE_CHECKING, Callable

import numpy as np

from manim.mobject.mobject import Mobject
from manim.mobject.opengl.opengl_mobject import OpenGLMobject

if TYPE_CHECKING:
    from manim.animation.protocol import AnimationProtocol


def assert_is_mobject_method(method: Callable) -> None:
    assert inspect.ismethod(method)
    mobject = method.__self__
    assert isinstance(mobject, (Mobject, OpenGLMobject))


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


def turn_animation_into_updater(
    animation: AnimationProtocol, cycle: bool = False, **kwargs
) -> Mobject:
    """
    Add an updater to the animation's mobject which applies
    the interpolation and update functions of the animation

    If cycle is True, this repeats over and over.  Otherwise,
    the updater will be popped upon completion

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
    animation.total_time = 0

    def update(m: Mobject, dt: float):
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


def cycle_animation(animation: AnimationProtocol, **kwargs) -> Mobject:
    return turn_animation_into_updater(animation, cycle=True, **kwargs)
