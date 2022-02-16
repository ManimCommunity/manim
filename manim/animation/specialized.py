from __future__ import annotations

__all__ = ["Broadcast"]

from typing import Any, Sequence

from manim.animation.transform import Restore

from ..constants import *
from .composition import LaggedStart


class Broadcast(LaggedStart):
    """Broadcast a mobject starting from an ``initial_width``, up to the actual size of the mobject.

    Parameters
    ----------
    mobject
        The mobject to be broadcast.
    focal_point
        The center of the broadcast, by default ORIGIN.
    n_mobs
        The number of mobjects that emerge from the focal point, by default 5.
    initial_opacity
        The starting stroke opacity of the mobjects emitted from the broadcast, by default 1.
    final_opacity
        The final stroke opacity of the mobjects emitted from the broadcast, by default 0.
    initial_width
        The initial width of the mobjects, by default 0.0.
    remover
        Whether the mobjects should be removed from the scene after the animation, by default True.
    lag_ratio
        The time between each iteration of the mobject, by default 0.2.
    run_time
        The total duration of the animation, by default 3.
    kwargs
        Additional arguments to be passed to :class:`~.LaggedStart`.

    Examples
    ---------

    .. manim:: BroadcastExample

        class BroadcastExample(Scene):
            def construct(self):
                mob = Circle(radius=4, color=TEAL_A)
                self.play(Broadcast(mob))
    """

    def __init__(
        self,
        mobject,
        focal_point: Sequence[float] = ORIGIN,
        n_mobs: int = 5,
        initial_opacity: float = 1,
        final_opacity: float = 0,
        initial_width: float = 0.0,
        remover: bool = True,
        lag_ratio: float = 0.2,
        run_time: float = 3,
        **kwargs: Any,
    ):
        self.focal_point = focal_point
        self.n_mobs = n_mobs
        self.initial_opacity = initial_opacity
        self.final_opacity = final_opacity
        self.initial_width = initial_width

        anims = []

        # Works by saving the mob that is passed into the animation, scaling it to 0 (or the initial_width) and then restoring the original mob.
        if mobject.fill_opacity:
            fill_o = True
        else:
            fill_o = False

        for _ in range(self.n_mobs):
            mob = mobject.copy()

            if fill_o:
                mob.set_opacity(self.final_opacity)
            else:
                mob.set_stroke(opacity=self.final_opacity)

            mob.move_to(self.focal_point)
            mob.save_state()
            mob.set_width(self.initial_width)

            if fill_o:
                mob.set_opacity(self.initial_opacity)
            else:
                mob.set_stroke(opacity=self.initial_opacity)

            anims.append(Restore(mob, remover=remover))

        super().__init__(*anims, run_time=run_time, lag_ratio=lag_ratio, **kwargs)
