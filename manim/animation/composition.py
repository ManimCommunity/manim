"""Tools for displaying multiple animations at once."""


from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from manim.mobject.opengl.opengl_mobject import OpenGLGroup

from .._config import config
from ..animation.animation import Animation, prepare_animation
from ..constants import RendererType
from ..mobject.mobject import Group, Mobject
from ..scene.scene import Scene
from ..utils.iterables import remove_list_redundancies
from ..utils.rate_functions import linear

if TYPE_CHECKING:
    from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVGroup

    from ..mobject.types.vectorized_mobject import VGroup

__all__ = ["AnimationGroup", "Succession", "LaggedStart", "LaggedStartMap"]


DEFAULT_LAGGED_START_LAG_RATIO: float = 0.05


class AnimationGroup(Animation):
    def __init__(
        self,
        *animations: Animation,
        group: Group | VGroup | OpenGLGroup | OpenGLVGroup = None,
        run_time: float | None = None,
        rate_func: Callable[[float], float] = linear,
        lag_ratio: float = 0,
        **kwargs,
    ) -> None:
        self.animations = [prepare_animation(anim) for anim in animations]
        self.rate_func = rate_func
        self.group = group
        if self.group is None:
            mobjects = remove_list_redundancies(
                [anim.mobject for anim in self.animations if not anim.is_introducer()],
            )
            if config["renderer"] == RendererType.OPENGL:
                self.group = OpenGLGroup(*mobjects)
            else:
                self.group = Group(*mobjects)
        super().__init__(
            self.group, rate_func=self.rate_func, lag_ratio=lag_ratio, **kwargs
        )
        self.run_time: float = self.init_run_time(run_time)

    def get_all_mobjects(self) -> Sequence[Mobject]:
        return list(self.group)

    def begin(self) -> None:
        if self.suspend_mobject_updating:
            self.group.suspend_updating()
        for anim in self.animations:
            anim.begin()

    def _setup_scene(self, scene) -> None:
        for anim in self.animations:
            anim._setup_scene(scene)

    def finish(self) -> None:
        for anim in self.animations:
            anim.finish()
        if self.suspend_mobject_updating:
            self.group.resume_updating()

    def clean_up_from_scene(self, scene: Scene) -> None:
        self._on_finish(scene)
        for anim in self.animations:
            if self.remover:
                anim.remover = self.remover
            anim.clean_up_from_scene(scene)

    def update_mobjects(self, dt: float) -> None:
        for anim in self.animations:
            anim.update_mobjects(dt)

    def init_run_time(self, run_time) -> float:
        self.build_animations_with_timings()
        if self.anims_with_timings:
            self.max_end_time = np.max([awt[2] for awt in self.anims_with_timings])
        else:
            self.max_end_time = 0
        return self.max_end_time if run_time is None else run_time

    def build_animations_with_timings(self) -> None:
        """
        Creates a list of triplets of the form
        (anim, start_time, end_time)
        """
        self.anims_with_timings = []
        curr_time: float = 0
        for anim in self.animations:
            start_time: float = curr_time
            end_time: float = start_time + anim.get_run_time()
            self.anims_with_timings.append((anim, start_time, end_time))
            # Start time of next animation is based on the lag_ratio
            curr_time = (1 - self.lag_ratio) * start_time + self.lag_ratio * end_time

    def interpolate(self, alpha: float) -> None:
        # Note, if the run_time of AnimationGroup has been
        # set to something other than its default, these
        # times might not correspond to actual times,
        # e.g. of the surrounding scene.  Instead they'd
        # be a rescaled version.  But that's okay!
        time = self.rate_func(alpha) * self.max_end_time
        for anim, start_time, end_time in self.anims_with_timings:
            anim_time = end_time - start_time
            if anim_time == 0:
                sub_alpha = 0
            else:
                sub_alpha = np.clip((time - start_time) / anim_time, 0, 1)
            anim.interpolate(sub_alpha)


class Succession(AnimationGroup):
    def __init__(self, *animations: Animation, lag_ratio: float = 1, **kwargs) -> None:
        super().__init__(*animations, lag_ratio=lag_ratio, **kwargs)

    def begin(self) -> None:
        assert len(self.animations) > 0
        self.update_active_animation(0)

    def finish(self) -> None:
        while self.active_animation is not None:
            self.next_animation()

    def update_mobjects(self, dt: float) -> None:
        if self.active_animation:
            self.active_animation.update_mobjects(dt)

    def _setup_scene(self, scene) -> None:
        if scene is None:
            return
        if self.is_introducer():
            for anim in self.animations:
                if not anim.is_introducer() and anim.mobject is not None:
                    scene.add(anim.mobject)

        self.scene = scene

    def update_active_animation(self, index: int) -> None:
        self.active_index = index
        if index >= len(self.animations):
            self.active_animation: Animation | None = None
            self.active_start_time: float | None = None
            self.active_end_time: float | None = None
        else:
            self.active_animation = self.animations[index]
            self.active_animation._setup_scene(self.scene)
            self.active_animation.begin()
            self.active_start_time = self.anims_with_timings[index][1]
            self.active_end_time = self.anims_with_timings[index][2]

    def next_animation(self) -> None:
        if self.active_animation is not None:
            self.active_animation.finish()
        self.update_active_animation(self.active_index + 1)

    def interpolate(self, alpha: float) -> None:
        current_time = self.rate_func(alpha) * self.max_end_time
        while self.active_end_time is not None and current_time >= self.active_end_time:
            self.next_animation()
        if self.active_animation is not None and self.active_start_time is not None:
            elapsed = current_time - self.active_start_time
            active_run_time = self.active_animation.get_run_time()
            subalpha = elapsed / active_run_time if active_run_time != 0.0 else 1.0
            self.active_animation.interpolate(subalpha)


class LaggedStart(AnimationGroup):
    def __init__(
        self,
        *animations: Animation,
        lag_ratio: float = DEFAULT_LAGGED_START_LAG_RATIO,
        **kwargs,
    ):
        super().__init__(*animations, lag_ratio=lag_ratio, **kwargs)


class LaggedStartMap(LaggedStart):
    def __init__(
        self,
        AnimationClass: Callable[..., Animation],
        mobject: Mobject,
        arg_creator: Callable[[Mobject], str] = None,
        run_time: float = 2,
        **kwargs,
    ) -> None:
        args_list = []
        for submob in mobject:
            if arg_creator:
                args_list.append(arg_creator(submob))
            else:
                args_list.append((submob,))
        anim_kwargs = dict(kwargs)
        if "lag_ratio" in anim_kwargs:
            anim_kwargs.pop("lag_ratio")
        animations = [AnimationClass(*args, **anim_kwargs) for args in args_list]
        super().__init__(*animations, run_time=run_time, **kwargs)
