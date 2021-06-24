from typing import Sequence

from colour import Color

from manim.animation.transform import Restore
from manim.mobject.mobject import Mobject

from ..constants import *
from ..mobject.geometry import Circle
from ..mobject.types.vectorized_mobject import VGroup
from ..utils.color import BLACK, WHITE
from .composition import LaggedStart


class Broadcast(LaggedStart):
    """Broadcast a mobject starting from an ``initial_width``, up to the actual size of the mobject.

    Parameters
    ----------
    mobject
        The mobject to be broadcast.
    focal_point
        The origin point of the broadcast, by default ORIGIN.
    n_mobs
        The number of mobjects that appear from the focal point, by default 5.
    initial_color
        The starting color of mobjects emitted from the broadcast, by default WHITE.
    final_color
        The final color of mobjects emitted from the broadcast, by default BLACK.
    initial_width
        The initial width of the mobjects, by default 0.0.
    remover
        Whether the given mobject should be removed from the scene after this animation, by default True.
    lag_raito
        The time between each appearance of the mobject, by default 0.2.
    run_time
        The total length of the animation, by default 3.
    kwargs
        Additional arguments to be passed to :class:`~.LaggedStart`.
    """

    def __init__(
        self,
        mobject: "Mobject",
        focal_point: Sequence[float] = ORIGIN,
        n_mobs: int = 5,
        initial_color: Color = WHITE,
        final_color: Color = BLACK,
        initial_width: float = 0.0,
        remover: bool = True,
        lag_raito: float = 0.2,
        run_time: float = 3,
        **kwargs
    ):
        self.focal_point = focal_point
        self.initial_width = initial_width
        self.n_mobs = n_mobs

        mobjects = VGroup(
            *[mobject.copy().set_stroke(final_color) for _ in range(self.n_mobs)]
        )
        for mobject in mobjects:
            mobject.add_updater(lambda c: c.move_to(self.focal_point))
            mobject.save_state()
            mobject.set(width=self.initial_width)
            mobject.set_stroke(color=initial_color)

        animations = [Restore(mobject) for mobject in mobjects]
        super().__init__(
            run_time=run_time,
            lag_ratio=lag_raito,
            remover=remover,
            *animations,
            **kwargs
        )
