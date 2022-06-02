import numpy as np

from typing import Callable, Dict
from manim.scene.scene import Scene

from ..animation.animation import Animation, Wait
from ..mobject.mobject import _AnimationBuilder


class ChangeSpeed(Animation):

    """
    Modifies the speed of passed animation. :class:`AnimationGroup` with
    different ``lag_ratio`` can also be used which combines multiple
    animations into one. `run_time` of the animation is changed to modify
    the speed.

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
            b = Dot().shift(LEFT * -4)
            self.add(a, b)
            self.play(
                ChangeSpeed(
                    anim=AnimationGroup(
                        a.animate(run_time=2).shift(RIGHT * 8),
                        b.animate(run_time=1).shift(LEFT * 8),
                    ),
                    speedinfo={0: 0.5, 1: 1},
                    rate_func=linear,
                )
            )

    """

    t = 0
    dt = 0

    def __init__(
        self,
        anim: Animation | _AnimationBuilder,
        speedinfo: Dict[float, float],
        rate_func: Callable[[float], float] = None,
        **kwargs,
    ) -> None:
        self.anim = anim.build() if type(anim) is _AnimationBuilder else anim
        if type(anim) is Wait:
            self.anim = ChangedWait(
                run_time=anim.run_time,
                stop_condition=anim.stop_condition,
                frozen_frame=anim.is_static_wait,
                **kwargs,
            )

        self.rate_func = self.anim.rate_func if rate_func is None else rate_func

        # Vertical parabola, f(0) = 0, f'(0) = m, f'( f-1(1) ) = n
        self.speed_modifier = lambda x, m, n: (n * n - m * m) * x * x / 4 + m * x

        # f-1(1), returns x where f(x) = 1 for the `speed_modifier` function
        self.f_inv_1 = lambda m, n: 2 / (m + n)

        if 0 not in speedinfo:
            speedinfo[0] = 1
        if 1 not in speedinfo:
            speedinfo[1] = sorted(speedinfo.items())[-1][1]

        self.speedinfo = dict(sorted(speedinfo.items()))
        self.functions = []
        self.conditions = []

        # To the total time
        total_time = self.get_total_time()

        prevnode = 0
        m = self.speedinfo[0]
        curr_time = 0
        for node, n in list(self.speedinfo.items())[1:]:
            dur = node - prevnode
            self.conditions.append(
                lambda x, curr_time=curr_time, m=m, n=n, dur=dur: curr_time / total_time
                <= x
                <= (curr_time + self.f_inv_1(m, n) * dur) / total_time
            )
            self.functions.append(
                lambda x, dur=dur, m=m, n=n, prevnode=prevnode, curr_time=curr_time: self.speed_modifier(
                    (total_time * x - curr_time) / dur, m, n
                )
                * dur
                + prevnode
            )
            curr_time += self.f_inv_1(m, n) * dur
            prevnode = node
            m = n

        def func(x):
            newx = np.piecewise(
                self.rate_func(x),
                [condition(self.rate_func(x)) for condition in self.conditions],
                self.functions,
            )
            ChangeSpeed.dt = (newx - self.t) * self.anim.run_time
            self.t = newx
            return newx

        self.anim.rate_func = func

        super().__init__(
            self.anim.mobject,
            rate_func=self.rate_func,
            run_time=total_time * self.anim.run_time,
            **kwargs,
        )

    def get_total_time(self) -> float:
        prevnode = 0
        m = self.speedinfo[0]
        total_time = 0
        for node, n in list(self.speedinfo.items())[1:]:
            dur = node - prevnode
            total_time += dur * self.f_inv_1(m, n)
            prevnode = node
            m = n
        # print(total_time)
        return total_time

    def interpolate(self, alpha: float) -> None:
        return self.anim.interpolate(alpha)

    def update_mobjects(self, dt: float) -> None:
        self.anim.update_mobjects(dt)

    def finish(self) -> None:
        self.anim.finish()
        if self.suspend_mobject_updating:
            self.anim.mobject.resume_updating()

    def begin(self) -> None:
        if self.suspend_mobject_updating:
            self.mobject.suspend_updating()
        self.anim.begin()

    def clean_up_from_scene(self, scene: Scene) -> None:
        self._on_finish(scene)
        if self.remover:
            self.anim.remover = self.remover
        self.anim.clean_up_from_scene(scene)


class ChangedWait(Wait):
    def __init__(
        self,
        run_time: float = 1,
        stop_condition: Callable[[], bool] | None = None,
        frozen_frame: bool | None = None,
        **kwargs,
    ):
        super().__init__(
            run_time=run_time,
            stop_condition=stop_condition,
            frozen_frame=frozen_frame,
            **kwargs,
        )

    def interpolate(self, alpha: float) -> None:
        self.get_sub_alpha(alpha, 0, 0)
