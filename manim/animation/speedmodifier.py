from __future__ import annotations

from typing import Callable, Optional

from numpy import piecewise

from manim.utils.simple_functions import get_parameters

from ..animation.animation import Animation, Wait, prepare_animation
from ..animation.composition import AnimationGroup
from ..mobject.mobject import Mobject, Updater, _AnimationBuilder
from ..scene.scene import Scene
from ..utils.rate_functions import linear


class ChangeSpeed(Animation):
    """
    Modifies the speed of passed animation. :class:`AnimationGroup` with
    different ``lag_ratio`` can also be used which combines multiple
    animations into one. `run_time` of the passed animation is changed to
    modify the speed.

    Parameters
    ----------
    anim : :class:`Animation` | :class:`_AnimationBuilder`
        Animation of which the speed is to be modified.
    speedinfo : Dict[float, float]
        Contains nodes (percentage of run_time) and its corresponding speed factor.
    rate_func : Callable[[float], float]
        Overrides `rate_func` of passed animation, applied before changing speed.

    Examples
    --------

    .. manim::SpeedModiferExample

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

    .. manim::SpeedModiferUpdaterExample

        class SpeedModifierUpdaterExample(Scene):
            def construct(self):
                a = Dot().shift(LEFT * 4)
                self.add(a)

                ChangeSpeed.add_updater(a, lambda x, dt: x.shift(RIGHT * 4 * dt))
                self.play(
                    ChangeSpeed(
                        Wait(2),
                        speedinfo={0.4: 1, 0.5: 0.2, 0.8: 0.2, 1: 1},
                    )
                )

    .. manim::SpeedModiferUpdaterExample2

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
                    )
                )

    """

    t = 0
    dt = 0
    changed = False

    def __init__(
        self,
        anim: Animation | _AnimationBuilder,
        speedinfo: dict[float, float],
        rate_func: Callable[[float], float] | None = None,
        **kwargs,
    ) -> None:

        self.anim = self.setup(anim)
        if issubclass(type(anim), AnimationGroup):
            self.anim = AnimationGroup(
                *map(self.setup, anim.animations),
                group=anim.group,
                run_time=anim.run_time,
                rate_func=anim.rate_func,
                lag_ratio=anim.lag_ratio,
            )

        self.rate_func = self.anim.rate_func if rate_func is None else rate_func

        # A function where, f(0) = 0, f'(0) = m, f'( f-1(1) ) = n
        # m being initial speed, n being final speed
        # Following function obtained when conditions applied to vertical parabola
        self.speed_modifier = (
            lambda x, initial_speed, final_speed: (
                final_speed**2 - initial_speed**2
            )
            * x**2
            / 4
            + initial_speed * x
        )

        # f-1(1), returns x for which f(x) = 1 in `speed_modifier` function
        self.f_inv_1 = lambda initial_speed, final_speed: 2 / (
            initial_speed + final_speed
        )

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
        m = self.speedinfo[0]
        curr_time = 0
        for node, n in list(self.speedinfo.items())[1:]:
            dur = node - prevnode
            self.conditions.append(
                lambda x, curr_time=curr_time, m=m, n=n, dur=dur: curr_time
                / scaled_total_time
                <= x
                <= (curr_time + self.f_inv_1(m, n) * dur) / scaled_total_time
            )
            self.functions.append(
                lambda x, dur=dur, m=m, n=n, prevnode=prevnode, curr_time=curr_time: self.speed_modifier(
                    (scaled_total_time * x - curr_time) / dur, m, n
                )
                * dur
                + prevnode
            )
            curr_time += self.f_inv_1(m, n) * dur
            prevnode = node
            m = n

        def func(t):
            new_t = piecewise(
                self.rate_func(t),
                [condition(self.rate_func(t)) for condition in self.conditions],
                self.functions,
            )
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
            return ChangedWait(
                run_time=anim.run_time,
                stop_condition=anim.stop_condition,
                frozen_frame=anim.is_static_wait,
            )
        return prepare_animation(anim)

    # Time taken by the animation if `run_time` is assumed to be 1
    def get_scaled_total_time(self) -> float:
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
        self,
        mobject: Mobject,
        update_function: Updater,
        index: int | None = None,
        call_updater: bool = False,
    ):
        parameters = get_parameters(update_function)
        if "dt" in parameters:
            mobject.add_updater(
                lambda m, dt: update_function(
                    m, ChangeSpeed.dt if ChangeSpeed.changed else dt
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
        ChangeSpeed.changed = False
        self.anim.finish()

    def begin(self) -> None:
        ChangeSpeed.changed = True
        self.anim.begin()

    def clean_up_from_scene(self, scene: Scene) -> None:
        self.anim.clean_up_from_scene(scene)

    def _setup_scene(self, scene) -> None:
        self.anim._setup_scene(scene)


class ChangedWait(Wait):
    """
    Wait animation but follows `rate_func`
    """

    def __init__(
        self,
        run_time: float = 1,
        stop_condition: Callable[[], bool] | None = None,
        frozen_frame: bool | None = None,
        rate_func: Callable[[float], float] = linear,
        **kwargs,
    ):
        super().__init__(
            run_time=run_time,
            rate_func=rate_func,
            stop_condition=stop_condition,
            frozen_frame=frozen_frame,
            **kwargs,
        )

    def interpolate(self, alpha: float) -> None:
        self.get_sub_alpha(alpha, 0, 0)
