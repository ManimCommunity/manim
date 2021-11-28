"""Animations that grow mobjects."""

__all__ = [
    "GrowFromPoint",
    "GrowFromCenter",
    "GrowFromEdge",
    "GrowArrow",
    "SpinInFromNothing",
]

import typing

import numpy as np

from ..animation.transform import Transform
from ..constants import PI
from ..utils.deprecation import deprecated_params
from ..utils.paths import spiral_path

if typing.TYPE_CHECKING:
    from ..mobject.geometry import Arrow
    from ..mobject.mobject import Mobject


class GrowFromPoint(Transform):
    def __init__(
        self, mobject: "Mobject", point: np.ndarray, point_color: str = None, **kwargs
    ) -> None:
        self.point = point
        self.point_color = point_color
        super().__init__(mobject, **kwargs)

    def create_target(self) -> "Mobject":
        return self.mobject

    def create_starting_mobject(self) -> "Mobject":
        start = super().create_starting_mobject()
        start.scale(0)
        start.move_to(self.point)
        if self.point_color:
            start.set_color(self.point_color)
        return start


class GrowFromCenter(GrowFromPoint):
    def __init__(self, mobject: "Mobject", point_color: str = None, **kwargs) -> None:
        point = mobject.get_center()
        super().__init__(mobject, point, point_color=point_color, **kwargs)


class GrowFromEdge(GrowFromPoint):
    def __init__(
        self, mobject: "Mobject", edge: np.ndarray, point_color: str = None, **kwargs
    ) -> None:
        point = mobject.get_critical_point(edge)
        super().__init__(mobject, point, point_color=point_color, **kwargs)


class GrowArrow(GrowFromPoint):
    def __init__(self, arrow: "Arrow", point_color: str = None, **kwargs) -> None:
        point = arrow.get_start()
        super().__init__(arrow, point, point_color=point_color, **kwargs)

    def create_starting_mobject(self) -> "Mobject":
        start_arrow = self.mobject.copy()
        start_arrow.scale(0, scale_tips=True, about_point=self.point)
        if self.point_color:
            start_arrow.set_color(self.point_color)
        return start_arrow


class SpinInFromNothing(GrowFromCenter):
    @deprecated_params(
        params="path_arc",
        message="Parameter angle is more robust, supporting angles greater than PI.",
    )
    def __init__(
        self,
        mobject: "Mobject",
        angle: float = PI / 2,
        point_color: str = None,
        **kwargs
    ) -> None:
        if "path_arc" in kwargs:
            super().__init__(mobject, point_color=point_color, **kwargs)
        else:
            self.angle = angle
            super().__init__(
                mobject,
                path_func=spiral_path(angle, mobject.get_center()),
                point_color=point_color,
                **kwargs
            )
