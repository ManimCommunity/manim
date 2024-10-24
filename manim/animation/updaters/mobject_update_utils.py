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
    from typing_extensions import Any, TypeVar

    from manim.animation.animation import Animation

    MobjectT = TypeVar("MobjectT", bound=Mobject)


def assert_is_mobject_method(method: Callable[[MobjectT], None]) -> None:
    """Verify that the given `method` is actually a method and belongs to a
    :class:`Mobject` or an :class:`OpenGLMobject`.

    Parameters
    ----------
    method
        An object which should be a method of :class:`Mobject` or :class:`OpenGLMobject`.

    Raises
    ------
    AssertionError
        If `method` is not a method or it doesn't belong to :class:`Mobject`
        or :class:`OpenGLMobject`.
    """
    assert inspect.ismethod(method)
    mobject = method.__self__
    assert isinstance(mobject, (Mobject, OpenGLMobject))


def always(method: Callable[[MobjectT], None], *args: Any, **kwargs) -> MobjectT:
    r"""Given the `method` of an existing :class:`Mobject`, apply an updater to
    this Mobject which modifies it on every frame by repeatedly calling the method.
    Additional arguments, both positional (`args`) and keyword arguments (`kwargs`),
    may be passed as arguments to `always`.

    Calling `always(mob.method, ...)` is equivalent to calling
    `mob.add_updater(lambda mob: mob.method(...))`.

    Parameters
    ----------
    method
        A Mobject method to call on each frame.
    args
        Positional arguments to be passed to `method`.
    kwargs
        Keyword arguments to be passed to `method`.

    Returns
    -------
    :class:`Mobject`
        The same Mobject whose `method` was passed to `always`, after adding an updater
        which repeatedly calls that method.

    Raises
    ------
    AssertionError
        If `method` is not a method or it doesn't belong to :class:`Mobject`
        or :class:`OpenGLMobject`.

    Examples
    --------

    .. manim:: AlwaysUpdatedSquares

        class AlwaysUpdatedSquares(Scene):
            def construct(self):
                dot_1 = Dot(color=RED).shift(3*LEFT)
                dot_2 = Dot(color=BLUE).shift(4*RIGHT)
                sq_1 = Square(color=RED)
                sq_2 = Square(color=BLUE)
                text_1 = Text(
                    "always(sq_1.next_to, dot_1, DOWN)",
                    font="Monospace",
                    color=RED_A,
                ).scale(0.4).move_to(2*UP + 4*LEFT)
                text_2 = Text(
                    "sq_2.add_updater(\n\tlambda mob: mob.next_to(dot_2, DOWN)\n)",
                    font="Monospace",
                    color=BLUE_A,
                ).scale(0.4).move_to(2*UP + 3*RIGHT)

                self.add(dot_1, dot_2, sq_1, sq_2, text_1, text_2)

                # Always place the squares below their respective dots.
                # The following two ways are equivalent.
                always(sq_1.next_to, dot_1, DOWN)
                sq_2.add_updater(lambda mob: mob.next_to(dot_2, DOWN))

                self.play(
                    dot_1.animate.shift(2*LEFT),
                    dot_2.animate.shift(2*LEFT),
                    run_time=2.0,
                )
    """
    assert_is_mobject_method(method)
    mobject = method.__self__
    func = method.__func__
    mobject.add_updater(lambda m: func(m, *args, **kwargs))
    return mobject


def f_always(
    method: Callable[[MobjectT], None],
    *arg_generators: Callable[[Any], Any],
    **kwargs: Any,
) -> MobjectT:
    r"""More functional version of :meth:`always`, where instead
    of taking in `args`, it takes in functions which output
    the relevant arguments.

    Parameters
    ----------
    method
        A Mobject method to call on each frame.
    arg_generators
        Functions which, when called, return positional arguments to be passed to `method`.
    kwargs
        Keyword arguments to be passed to `method`.

    Returns
    -------
    :class:`Mobject`
        The same Mobject whose `method` was passed to `f_always`, after adding an updater
        which repeatedly calls that method.

    Raises
    ------
    AssertionError
        If `method` is not a method or it doesn't belong to :class:`Mobject`
        or :class:`OpenGLMobject`.

    Examples
    --------

    .. manim:: FAlwaysUpdatedSquares

        class FAlwaysUpdatedSquares(Scene):
            def construct(self):
                sq_1 = Square(color=RED).move_to(DOWN + 4*LEFT)
                sq_2 = Square(color=BLUE).move_to(DOWN + 3*RIGHT)
                text_1 = Text(
                    "f_always(sq_1.set_opacity, t.get_value)",
                    font="Monospace",
                    color=RED_A,
                ).scale(0.35).move_to(UP + 4*LEFT)
                text_2 = Text(
                    "sq_2.add_updater(\n\tlambda mob: mob.set_opacity(t.get_value())\n)",
                    font="Monospace",
                    color=BLUE_A,
                ).scale(0.35).move_to(UP + 3*RIGHT)

                self.add(sq_1, sq_2, text_1, text_2)

                t = ValueTracker(1.0)

                # Always set the square opacities to the value given by t.
                # The following two ways are equivalent.
                f_always(sq_1.set_opacity, t.get_value)
                sq_2.add_updater(
                    lambda mob: mob.set_opacity(t.get_value())
                )

                self.play(
                    t.animate.set_value(0.1),
                    run_time=2.0,
                )
    """
    assert_is_mobject_method(method)
    mobject = method.__self__
    func = method.__func__

    def updater(mob):
        args = [arg_generator() for arg_generator in arg_generators]
        func(mob, *args, **kwargs)

    mobject.add_updater(updater)
    return mobject


def always_redraw(func: Callable[[], MobjectT]) -> MobjectT:
    """Redraw the Mobject constructed by a function every frame.

    This function returns a Mobject with an attached updater that
    continuously regenerates the mobject according to the
    specified function.

    Parameters
    ----------
    func
        A function without (required) input arguments that returns
        a Mobject.

    Returns
    -------
    :class:`Mobject`
        The Mobject returned by the function, after adding an updater to it
        which constantly transforms it, according to the given `func`.

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
    mobject: MobjectT, direction: np.ndarray[np.float64] = RIGHT, rate: float = 0.1
) -> MobjectT:
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

    Returns
    -------
    :class:`Mobject`
        The same Mobject, after adding an updater to it which shifts it continuously.

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


def always_rotate(mobject: MobjectT, rate: float = 20 * DEGREES, **kwargs) -> MobjectT:
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

    Returns
    -------
    :class:`Mobject`
        The same Mobject, after adding an updater which rotates it continuously.

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


def turn_animation_into_updater(animation: Animation, cycle: bool = False) -> Mobject:
    """Add an updater to the animation's Mobject, which applies
    the interpolation and update functions of the animation.

    If `cycle` is `True`, this repeats over and over. Otherwise,
    the updater will be popped upon completion.

    Parameters
    ----------
    animation
        The animation to convert into an updater.
    cycle
        Whether to repeat the animation over and over, or do it
        only once and remove the updater once finished.

    Returns
    -------
    :class:`Mobject`
        The Mobject being modified by the original `animation` which
        was converted into an updater for this Mobject.

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


def cycle_animation(animation: Animation) -> Mobject:
    """Same as `turn_animation_into_updater`, but with `cycle=True`.

    Parameters
    ----------
    animation
        The animation to convert into an updater which will be repeated
        forever.

    Returns
    -------
    :class:`Mobject`
        The Mobject being modified by the original `animation` which
        was converted into an updater for this Mobject.
    """
    return turn_animation_into_updater(animation, cycle=True)
