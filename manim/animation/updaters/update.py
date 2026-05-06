"""Animations that update mobjects."""

from __future__ import annotations

__all__ = ["UpdateFromFunc", "UpdateFromAlphaFunc", "MaintainPositionRelativeTo"]


import operator as op
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from manim.animation.animation import Animation

if TYPE_CHECKING:
    from manim.mobject.mobject import Mobject


class UpdateFromFunc(Animation):
    """ Updates the mobject based on an update_function of the form func(mobject).
    
    This can be used when the state of one mobject is dependent
    on another simultaneously animated mobject.

    Parameters
    ----------
    mobject
        The mobject which needs to be updated.

    update_function
        The function of the form func(mobject) 
        which would determine how the mobject is getting updated each frame.
    
    suspend_mobject_updating
        If ``True``, any updater added via ``add_updater`` on the mobject
        is suspended for the duration of this animation. Defaults to ``False``.
    
    **kwargs
        Any other keyword arguments to be passed to :class:`Animation`.

    Example::

        from manim import *

        class UpdateFromFuncDemo(Scene):
            def construct(self):         
                dot = Dot().to_edge(1.5*LEFT)
                label = Text("Hello").next_to(dot,UP)

                def update_func(mob):
                    mob.next_to(dot,UP)

                self.play(
                    dot.animate.to_edge(1.5*RIGHT),
                    UpdateFromFunc(label, update_func),
                    run_time=3,
                )
    """

    def __init__(
        self,
        mobject: Mobject,
        update_function: Callable[[Mobject], Any],
        suspend_mobject_updating: bool = False,
        **kwargs: Any,
    ) -> None:
        self.update_function = update_function
        super().__init__(
            mobject, suspend_mobject_updating=suspend_mobject_updating, **kwargs
        )

    def interpolate_mobject(self, alpha: float) -> None:
        self.update_function(self.mobject)  # type: ignore[arg-type]


class UpdateFromAlphaFunc(UpdateFromFunc):
    """ Updates the mobject based on an update_function of the form func(mobject, alpha).
    
    This can be used when the state of one mobject is dependent on:
    (1) Another simultaneously animated mobject
    (2) alpha value

    Parameters
    ----------
    mobject
        The mobject which needs to be updated.

    update_function
        The function of the form func(mobject, alpha) 
        which would determine how the mobject is getting updated each frame.
    
    suspend_mobject_updating
        If ``True``, any updater added via ``add_updater`` on the mobject
        is suspended for the duration of this animation. Defaults to ``False``.
    
    **kwargs
        Any other keyword arguments to be passed to :class:`Animation`.

    Example::

        from manim import *

        class UpdateFromAlphaFuncDemo(Scene):
            def construct(self):         
                dot = Dot().to_edge(1.5*LEFT)
                label = Text("Hello").next_to(dot,UP)
                number = DecimalNumber()        
                vg = VGroup(label, number)
                self.add(vg)

                def update_func(mob, alpha):
                    m,n = mob
                    m.next_to(dot,UP)
                    m.set_opacity(alpha)
                    n.set_value(alpha).next_to(dot, DOWN)

                self.play(
                    dot.animate.to_edge(1.5*RIGHT),
                    UpdateFromAlphaFunc(vg, update_func),
                    run_time=3,
                )
    """
    def interpolate_mobject(self, alpha: float) -> None:
        self.update_function(self.mobject, self.rate_func(alpha))  # type: ignore[call-arg, arg-type]


class MaintainPositionRelativeTo(Animation):
    """ Useful when one mobject's position is to be maintained constant w.r.t another mobject's position.

    Parameters
    ----------
    mobject
        The mobject whose position is to be kept constant w.r.t another mobject
    tracked_mobject
        This is the mobject w.r.t whose position, the mobject's position is to be kept constant.
    **kwargs
        Any other keyword arguments to be passed to :class:`Animation`.
    
    Example::

        from manim import *

        class MaintainPositionRelativeToDemo(Scene):
            def construct(self):         
                dot = Dot().to_edge(1.5*LEFT)
                label = Text("Hello").next_to(dot,UP)

                self.play(
                    dot.animate.to_edge(1.5*RIGHT),
                    MaintainPositionRelativeTo(label, dot),
                    run_time=3,
                )

    """
    def __init__(
        self, mobject: Mobject, tracked_mobject: Mobject, **kwargs: Any
    ) -> None:
        self.tracked_mobject = tracked_mobject
        self.diff = op.sub(
            mobject.get_center(),
            tracked_mobject.get_center(),
        )
        super().__init__(mobject, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        target = self.tracked_mobject.get_center()
        location = self.mobject.get_center()
        self.mobject.shift(target + self.diff - location)
