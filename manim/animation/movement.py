"""Animations related to movement."""

__all__ = [
    "Homotopy",
    "SmoothedVectorizedHomotopy",
    "ComplexHomotopy",
    "PhaseFlow",
    "MoveAlongPath",
]

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import numpy as np

from ..animation.animation import Animation
from ..utils.rate_functions import linear

if TYPE_CHECKING:
    from ..mobject.mobject import Mobject


class Homotopy(Animation):
    def __init__(
        self,
        homotopy: Callable[[float, float, float, float], Tuple[float, float, float]],
        mobject: "Mobject",
        run_time: float = 3,
        apply_function_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Homotopy is a function from
        (x, y, z, t) to (x', y', z')
        """
        self.homotopy = homotopy
        self.apply_function_kwargs = (
            apply_function_kwargs if apply_function_kwargs is not None else {}
        )
        super().__init__(mobject, run_time=run_time, **kwargs)

    def function_at_time_t(self, t: float) -> Tuple[float, float, float]:
        return lambda p: self.homotopy(*p, t)

    def interpolate_submobject(
        self,
        submobject: "Mobject",
        starting_submobject: "Mobject",
        alpha: float,
    ) -> None:
        submobject.points = starting_submobject.points
        submobject.apply_function(
            self.function_at_time_t(alpha), **self.apply_function_kwargs
        )


class SmoothedVectorizedHomotopy(Homotopy):
    def interpolate_submobject(
        self,
        submobject: "Mobject",
        starting_submobject: "Mobject",
        alpha: float,
    ) -> None:
        super().interpolate_submobject(submobject, starting_submobject, alpha)
        submobject.make_smooth()


class ComplexHomotopy(Homotopy):
    def __init__(
        self, complex_homotopy: Callable[[complex], float], mobject: "Mobject", **kwargs
    ) -> None:
        """
        Complex Homotopy a function Cx[0, 1] to C
        """

        def homotopy(
            x: float,
            y: float,
            z: float,
            t: float,
        ) -> Tuple[float, float, float]:
            c = complex_homotopy(complex(x, y), t)
            return (c.real, c.imag, z)

        super().__init__(homotopy, mobject, **kwargs)


class PhaseFlow(Animation):
    def __init__(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        mobject: "Mobject",
        virtual_time: float = 1,
        suspend_mobject_updating: bool = False,
        rate_func: Callable[[float], float] = linear,
        **kwargs
    ) -> None:
        self.virtual_time = virtual_time
        self.function = function
        super().__init__(
            mobject,
            suspend_mobject_updating=suspend_mobject_updating,
            rate_func=rate_func,
            **kwargs
        )

    def interpolate_mobject(self, alpha: float) -> None:
        if hasattr(self, "last_alpha"):
            dt = self.virtual_time * (
                self.rate_func(alpha) - self.rate_func(self.last_alpha)
            )
            self.mobject.apply_function(lambda p: p + dt * self.function(p))
        self.last_alpha = alpha


class MoveAlongPath(Animation):
    """Make one mobject move along the path of another mobject.
    Example
    --------
    .. manim:: MoveAlongPathExample

        class MoveAlongPathExample(Scene):
            def construct(self):
                d1 = Dot().set_color(ORANGE)
                l1 = Line(LEFT, RIGHT)
                l2 = VMobject()
                self.add(d1, l1, l2)
                l2.add_updater(lambda x: x.become(Line(LEFT, d1.get_center()).set_color(ORANGE)))
                self.play(MoveAlongPath(d1, l1), rate_func=linear)
    """

    def __init__(
        self,
        mobject: "Mobject",
        path: np.ndarray,
        suspend_mobject_updating: Optional[bool] = False,
        **kwargs
    ) -> None:
        self.path = path
        super().__init__(
            mobject, suspend_mobject_updating=suspend_mobject_updating, **kwargs
        )

    def interpolate_mobject(self, alpha: float) -> None:
        point = self.path.point_from_proportion(self.rate_func(alpha))
        self.mobject.move_to(point)
