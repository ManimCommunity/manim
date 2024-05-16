"""Utilities for modifying the speed at which animations are played."""

from __future__ import annotations

import inspect
import types
from typing import TYPE_CHECKING, Callable

from numpy import piecewise

from ..animation.animation import Animation, Wait, prepare_animation
from ..animation.composition import AnimationGroup
from ..mobject.mobject import Mobject, _AnimationBuilder
from ..scene.scene import Scene

if TYPE_CHECKING:
    from ..mobject.mobject import Updater

__all__ = ["ChangeSpeed"]


class ChangeSpeed(Animation):
    """Modifies the speed of passed animation.
    :class:`AnimationGroup` with different ``lag_ratio`` can also be used
    which combines multiple animations into one.
    The ``run_time`` of the passed animation is changed to modify the speed.

    Parameters
    ----------
    anim
        Animation of which the speed is to be modified.
    speedinfo
        Contains nodes (percentage of ``run_time``) and its corresponding speed factor.
    rate_func
        Overrides ``rate_func`` of passed animation, applied before changing speed.

    Examples
    --------

    .. manim:: SpeedModifierExample

        class SpeedModifierExample(Scene):
            def construct(self):
                a = Dot().shift(LEFT * 4)
                b = Dot().shift(RIGHT * 4)
                self.add(a, b)
                self.play(
                    ChangeSpeed(
                        AnimationGroup(
                            a.animate(run_time=1).shift(RIGHT * 8),
                            b.animate(run_time=1).shift(LEFT * 8),
                        ),
                        speedinfo={0.3: 1, 0.4: 0.1, 0.6: 0.1, 1: 1},
                        rate_func=linear,
                    )
                )

    .. manim:: SpeedModifierUpdaterExample

        class SpeedModifierUpdaterExample(Scene):
            def construct(self):
                a = Dot().shift(LEFT * 4)
                self.add(a)

                ChangeSpeed.add_updater(a, lambda x, dt: x.shift(RIGHT * 4 * dt))
                self.play(
                    ChangeSpeed(
                        Wait(2),
                        speedinfo={0.4: 1, 0.5: 0.2, 0.8: 0.2, 1: 1},
                        affects_speed_updaters=True,
                    )
                )

    .. manim:: SpeedModifierUpdaterExample2

        class SpeedModifierUpdaterExample2(Scene):
            def construct(self):
                a = Dot().shift(LEFT * 4)
                self.add(a)

                ChangeSpeed.add_updater(a, lambda x, dt: x.shift(RIGHT * 4 * dt))
                self.wait()
                self.play(
                    ChangeSpeed(
                        Wait(),
                        speedinfo={1: 0},
                        affects_speed_updaters=True,
                    )
                )

    """

    dt = 0
    is_changing_dt = False

    def __init__(
        self,
        anim: Animation | _AnimationBuilder,
        speedinfo: dict[float, float],
        rate_func: Callable[[float], float] | None = None,
        affects_speed_updaters: bool = True,
        **kwargs,
    ) -> None:
        if issubclass(type(anim), AnimationGroup):
            self.anim = type(anim)(
                *map(self.setup, anim.animations),
                group=anim.group,
                run_time=anim.run_time,
                rate_func=anim.rate_func,
                lag_ratio=anim.lag_ratio,
            )
        else:
            self.anim = self.setup(anim)

        if affects_speed_updaters:
            assert (
                ChangeSpeed.is_changing_dt is False
            ), "Only one animation at a time can play that changes speed (dt) for ChangeSpeed updaters"
            ChangeSpeed.is_changing_dt = True
            self.t = 0
        self.affects_speed_updaters = affects_speed_updaters

        self.rate_func = self.anim.rate_func if rate_func is None else rate_func

        # A function where, f(0) = 0, f'(0) = initial speed, f'( f-1(1) ) = final speed
        # Following function obtained when conditions applied to vertical parabola
        self.speed_modifier = lambda x, init_speed, final_speed: (
            (final_speed**2 - init_speed**2) * x**2 / 4 + init_speed * x
        )

        # f-1(1), returns x for which f(x) = 1 in `speed_modifier` function
        self.f_inv_1 = lambda init_speed, final_speed: 2 / (init_speed + final_speed)

        # if speed factors for the starting node (0) and the final node (1) are
        # not set, set them to 1 and the penultimate factor, respectively
        if 0 not in speedinfo:
            speedinfo[0] = 1
        if 1 not in speedinfo:
            speedinfo[1] = sorted(speedinfo.items())[-1][1]

        self.speedinfo = dict(sorted(speedinfo.items()))
        self.functions = []
        self.conditions = []

        # Get the time taken by amimation if `run_time` is assumed to be 1
        scaled_total_time = self.get_scaled_total_time()

        prevnode = 0
        init_speed = self.speedinfo[0]
        curr_time = 0
        for node, final_speed in list(self.speedinfo.items())[1:]:
            dur = node - prevnode

            def condition(
                t,
                curr_time=curr_time,
                init_speed=init_speed,
                final_speed=final_speed,
                dur=dur,
            ):
                lower_bound = curr_time / scaled_total_time
                upper_bound = (
                    curr_time + self.f_inv_1(init_speed, final_speed) * dur
                ) / scaled_total_time
                return lower_bound <= t <= upper_bound

            self.conditions.append(condition)

            def function(
                t,
                curr_time=curr_time,
                init_speed=init_speed,
                final_speed=final_speed,
                dur=dur,
                prevnode=prevnode,
            ):
                return (
                    self.speed_modifier(
                        (scaled_total_time * t - curr_time) / dur,
                        init_speed,
                        final_speed,
                    )
                    * dur
                    + prevnode
                )

            self.functions.append(function)

            curr_time += self.f_inv_1(init_speed, final_speed) * dur
            prevnode = node
            init_speed = final_speed

        def func(t):
            if t == 1:
                ChangeSpeed.is_changing_dt = False
            new_t = piecewise(
                self.rate_func(t),
                [condition(self.rate_func(t)) for condition in self.conditions],
                self.functions,
            )
            if self.affects_speed_updaters:
                ChangeSpeed.dt = (new_t - self.t) * self.anim.run_time
                self.t = new_t
            return new_t

        self.anim.set_rate_func(func)

        super().__init__(
            self.anim.mobject,
            rate_func=self.rate_func,
            run_time=scaled_total_time * self.anim.run_time,
            **kwargs,
        )

    def setup(self, anim):
        if type(anim) is Wait:
            anim.interpolate = types.MethodType(
                lambda self, alpha: self.rate_func(alpha), anim
            )
        return prepare_animation(anim)

    def get_scaled_total_time(self) -> float:
        """The time taken by the animation under the assumption that the ``run_time`` is 1."""
        prevnode = 0
        init_speed = self.speedinfo[0]
        total_time = 0
        for node, final_speed in list(self.speedinfo.items())[1:]:
            dur = node - prevnode
            total_time += dur * self.f_inv_1(init_speed, final_speed)
            prevnode = node
            init_speed = final_speed
        return total_time

    @classmethod
    def add_updater(
        cls,
        mobject: Mobject,
        update_function: Updater,
        index: int | None = None,
        call_updater: bool = False,
    ):
        """This static method can be used to apply speed change to updaters.

        This updater will follow speed and rate function of any :class:`.ChangeSpeed`
        animation that is playing with ``affects_speed_updaters=True``. By default,
        updater functions added via the usual :meth:`.Mobject.add_updater` method
        do not respect the change of animation speed.

        Parameters
        ----------
        mobject
            The mobject to which the updater should be attached.
        update_function
            The function that is called whenever a new frame is rendered.
        index
            The position in the list of the mobject's updaters at which the
            function should be inserted.
        call_updater
            If ``True``, calls the update function when attaching it to the
            mobject.

        See also
        --------
        :class:`.ChangeSpeed`
        :meth:`.Mobject.add_updater`
        """
        if "dt" in inspect.signature(update_function).parameters:
            mobject.add_updater(
                lambda mob, dt: update_function(
                    mob, ChangeSpeed.dt if ChangeSpeed.is_changing_dt else dt
                ),
                index=index,
                call_updater=call_updater,
            )
        else:
            mobject.add_updater(update_function, index=index, call_updater=call_updater)

    def interpolate(self, alpha: float) -> None:
        self.anim.interpolate(alpha)

    def update_mobjects(self, dt: float) -> None:
        self.anim.update_mobjects(dt)

    def finish(self) -> None:
        ChangeSpeed.is_changing_dt = False
        self.anim.finish()

    def begin(self) -> None:
        self.anim.begin()

    def clean_up_from_scene(self, scene: Scene) -> None:
        self.anim.clean_up_from_scene(scene)

    def _setup_scene(self, scene) -> None:
        self.anim._setup_scene(scene)
