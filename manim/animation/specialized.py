__all__ = ["Broadcast"]

from typing import Sequence

from manim.animation.transform import Restore
from manim.mobject.mobject import Mobject

from ..constants import *
from ..mobject.types.vectorized_mobject import VGroup
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
    initial_opacity
        The starting stroke opacity of mobjects emitted from the broadcast, by default 1.
    final_opacity
        The final stroke opacity of mobjects emitted from the broadcast, by default 0.
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
        mobject: "Mobject",
        focal_point: Sequence[float] = ORIGIN,
        n_mobs: int = 5,
        initial_opacity: float = 1,
        final_opacity: float = 0,
        initial_width: float = 0.0,
        remover: bool = True,
        lag_raito: float = 0.2,
        run_time: float = 3,
        **kwargs
    ):
        self.focal_point = focal_point
        self.n_mobs = n_mobs
        self.initial_opacity = initial_opacity
        self.final_opacity = final_opacity
        self.initial_width = initial_width

        # create all the mobjects and move them to the focal point
        mobjects = VGroup(
            *[
                mobject.copy()
                .set_stroke(opacity=self.final_opacity)
                .move_to(self.focal_point)
                for _ in range(self.n_mobs)
            ]
        )

        for mobject in mobjects:
            mobject.save_state()
            mobject.set(width=self.initial_width)
            mobject.set_stroke(opacity=self.initial_opacity)

        # restore the mob to its original status
        # to create the effect of it growing from nothing
        animations = [Restore(mobject) for mobject in mobjects]
        super().__init__(
            *animations,
            run_time=run_time,
            lag_ratio=lag_raito,
            remover=remover,
            **kwargs
        )
