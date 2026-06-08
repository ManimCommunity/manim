from typing import Any

import numpy as np

from manim.mobject.geometry.arc import Arc
from manim.mobject.types.vectorized_mobject import VMobject


class Polyline(VMobject):
    def __init__(
        self,
        points: list[tuple[float, float, float]],
        corner_radius: float = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.points_input = points
        self.corner_radius = corner_radius
        self._generate_points()

    def _generate_points(self) -> None:
        if self.corner_radius <= 0:
            self.set_points_as_corners(self.points_input)
        else:
            self._generate_rounded_points()

    def _generate_rounded_points(self) -> None:
        points = self.points_input
        radius = self.corner_radius
        new_points: list = []

        for i in range(len(points)):
            if i == 0 or i == len(points) - 1:
                new_points.append(points[i])
                continue

            A = np.array(points[i - 1])
            B = np.array(points[i])
            C = np.array(points[i + 1])

            BA = A - B
            BC = C - B

            BA_norm = BA / np.linalg.norm(BA)
            BC_norm = BC / np.linalg.norm(BC)

            angle = np.arccos(np.dot(BA_norm, BC_norm))
            d = radius / np.tan(angle / 2)

            d1 = min(d, np.linalg.norm(BA))
            start = B + BA_norm * d1

            if i == 1:
                new_points.append(start)
            else:
                new_points.append(start)

            cross = np.cross(BA_norm, BC_norm)
            sign = 1 if cross > 0 else -1

            arc = Arc(
                radius=radius,
                start_angle=np.arctan2(BA_norm[1], BA_norm[0]),
                angle=sign * angle,
                arc_center=B,
            )
            arc_points = arc.get_points()
            if len(arc_points) > 1:
                new_points.extend(arc_points[1:])

        self.set_points_as_corners(new_points)
