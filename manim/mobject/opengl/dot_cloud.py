from __future__ import annotations

__all__ = ["TrueDot", "DotCloud"]

from typing import Any, Self

import numpy as np

from manim.constants import ORIGIN, RIGHT, UP
from manim.mobject.opengl.opengl_point_cloud_mobject import OpenGLPMobject
from manim.typing import Point3DLike
from manim.utils.color import YELLOW, ParsableManimColor


class DotCloud(OpenGLPMobject):
    def __init__(
        self,
        color: ParsableManimColor = YELLOW,
        stroke_width: float = 2.0,
        radius: float = 2.0,
        density: float = 10,
        **kwargs: Any,
    ):
        self.radius = radius
        self.epsilon = 1.0 / density
        super().__init__(
            stroke_width=stroke_width, density=density, color=color, **kwargs
        )

    def init_points(self) -> None:
        self.points = np.array(
            [
                r * (np.cos(theta) * RIGHT + np.sin(theta) * UP)
                for r in np.arange(self.epsilon, self.radius, self.epsilon)
                # Num is equal to int(stop - start)/ (step + 1) reformulated.
                for theta in np.linspace(
                    0,
                    2 * np.pi,
                    num=int(2 * np.pi * (r + self.epsilon) / self.epsilon),
                )
            ],
            dtype=np.float32,
        )

    def make_3d(self, gloss: float = 0.5, shadow: float = 0.2) -> Self:
        self.set_gloss(gloss)
        self.set_shadow(shadow)
        self.apply_depth_test()
        return self


class TrueDot(DotCloud):
    def __init__(
        self, center: Point3DLike = ORIGIN, stroke_width: float = 2.0, **kwargs: Any
    ):
        self.radius = stroke_width
        super().__init__(points=[center], stroke_width=stroke_width, **kwargs)
