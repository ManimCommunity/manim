"""Utility functions for continuous animation of mobjects."""

from __future__ import annotations

__all__ = [
    "always",
    "f_always",
    "always_redraw",
    "turn_animation_into_updater",
    "cycle_animation",
]


import inspect
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

import numpy as np

from manim.mobject.opengl.opengl_mobject import OpenGLMobject

if TYPE_CHECKING:
    import types

    from typing_extensions import Concatenate, ParamSpec, TypeIs

    from manim.animation.protocol import MobjectAnimation

    P = ParamSpec("P")


M = TypeVar("M", bound=OpenGLMobject)


# TODO: figure out how to typehint as MethodType[OpenGLMobject] to avoid the cast
# madness in always/f_always
def is_mobject_method(method: Callable[..., Any]) -> TypeIs[types.MethodType]:
    return inspect.ismethod(method) and isinstance(method.__self__, OpenGLMobject)


def always(
    method: Callable[Concatenate[M, P], object], *args: P.args, **kwargs: P.kwargs
) -> M:
    if not is_mobject_method(method):
        raise ValueError("always must take a method of a Mobject")
    mobject = cast(M, method.__self__)
    func = method.__func__
    mobject.add_updater(lambda m: func(m, *args, **kwargs))
    return mobject


def f_always(
    method: Callable[Concatenate[M, ...], None],
    *arg_generators: Callable[[], object],
    **kwargs,
) -> M:
    """
    More functional version of always, where instead
    of taking in args, it takes in functions which output
    the relevant arguments.
    """
    if not is_mobject_method(method):
        raise ValueError("f_always must take a method of a Mobject")
    mobject = cast(M, method.__self__)
    func = method.__func__

    def updater(mob):
        args = [arg_generator() for arg_generator in arg_generators]
        func(mob, *args, **kwargs)

    mobject.add_updater(updater)
    return mobject


def always_redraw(func: Callable[[], M]) -> M:
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
    animation: MobjectAnimation[M], cycle: bool = False
) -> M:
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
    total_time = 0

    def update(m: OpenGLMobject, dt: float):
        nonlocal total_time
        run_time = animation.get_run_time()
        time_ratio = total_time / run_time
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
        total_time += dt

    mobject.add_updater(update)
    return mobject


def cycle_animation(animation: MobjectAnimation[M], **kwargs) -> M:
    return turn_animation_into_updater(animation, cycle=True, **kwargs)
