from cgi import test
from cloup import constrained_params
import numpy as np

from manim.scene.scene import Scene

from ..utils.rate_functions import linear
from ..utils.iterables import remove_list_redundancies
from ..animation.animation import Animation
from ..mobject.mobject import _AnimationBuilder, Group, Mobject
from ..mobject.opengl.opengl_mobject import OpenGLGroup
from .._config import config


class ChangeSpeed(Animation):
    def __init__(
        self,
        anim: Animation | _AnimationBuilder,
        speedinfo,
        **kwargs,
    ) -> None:
        self.anim = anim.build() if type(anim) is _AnimationBuilder else anim
        self.speed_modifier = lambda x, m, n: (n * n - m * m) * x * x / 4 + m * x
        self.f_inv_1 = lambda m, n: 2 / (m + n)

        if 0 not in speedinfo:
            speedinfo[0] = 1
        if 1 not in speedinfo:
            speedinfo[1] = sorted(speedinfo.items())[-1][1]

        self.speedinfo = dict(sorted(speedinfo.items()))
        self.run_time = 0
        self.functions = []
        self.conditions = []

        prevnode = 0
        m = self.speedinfo[0]
        total_time = 0
        for node, n in list(self.speedinfo.items())[1:]:
            dur = node - prevnode
            total_time += dur * self.f_inv_1(m, n)
            prevnode = node
            m = n

        prevnode = 0
        m = self.speedinfo[0]
        curr_time = 0
        offset = 0
        for node, n in list(self.speedinfo.items())[1:]:
            dur = node - prevnode
            self.conditions.append(
                lambda x, offset=offset, m=m, n=n, dur=dur: offset / total_time
                <= x
                <= (offset + self.f_inv_1(m, n) * dur) / total_time
            )
            self.functions.append(
                lambda x, dur=dur, m=m, n=n, prevnode=prevnode, total_time=total_time, offset=offset: self.speed_modifier(
                    (total_time * x - offset) / dur, m, n
                )
                * dur
                + prevnode
            )
            offset += self.f_inv_1(m, n) * dur
            curr_time += dur * self.f_inv_1(m, n)
            prevnode = node
            m = n

        def test(x):
            return np.piecewise(
                x, [condition(x) for condition in self.conditions], self.functions
            )

        self.anim.rate_func = test

        super().__init__(
            anim.mobject,
            run_time=total_time,
            **kwargs,
        )

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
