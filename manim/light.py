"""Lights: what lit materials (:class:`~.PbrMaterial`) are shaded by.

A light is a mobject: it is added to a scene, moved, animated and given updaters
like any other, and draws nothing. A frame without lights is not shaded at all: lit
materials show their base colour. :class:`.ThreeDScene` has one light,
``scene.light``, which is not in ``scene.mobjects`` (so :meth:`.Scene.clear` and
fading out every mobject leave it); lights added with :meth:`.Scene.add` are
ordinary mobjects of the scene.

Light units follow glTF ``KHR_lights_punctual``: a point light's intensity is in
candela and falls off with the square of the distance, and shading happens in
linear RGB. A white surface facing a light of intensity ``pi * d**2`` at distance
``d`` is fully lit.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from manim.constants import ORIGIN
from manim.mobject.mobject import Mobject
from manim.utils.bezier import interpolate
from manim.utils.color import WHITE, ManimColor, ParsableManimColor
from manim.utils.versioned_fields import Field

if TYPE_CHECKING:
    from typing import Self

    from manim.typing import Point3D, Point3DLike

__all__ = [
    "DEFAULT_AMBIENT_LIGHT",
    "DEFAULT_LIGHT_INTENSITY",
    "DEFAULT_LIGHT_POSITION",
    "PointLight",
]

#: A white surface facing the light at the default position (``sqrt(230)`` away)
#: is fully lit.
DEFAULT_LIGHT_INTENSITY = math.pi * 230.0
#: The intensity of the white light that reaches every surface
#: (:attr:`.Scene.ambient_light`).
DEFAULT_AMBIENT_LIGHT = 0.25
#: Where :class:`.ThreeDScene` puts its light.
DEFAULT_LIGHT_POSITION = np.array([-7.0, -9.0, 10.0])


class PointLight(Mobject):
    """A light at a point, shining equally in all directions.

    Parameters
    ----------
    position
        Where the light is.
    color
        Its colour.
    intensity
        Its intensity in candela (see :mod:`manim.light`).

    Examples
    --------
    .. code-block:: python

        class Lit(ThreeDScene):
            def construct(self):
                self.add(Sphere())
                self.play(self.light.animate.move_to(3 * RIGHT))
    """

    light_color: Field[ManimColor] = Field()
    intensity: Field[float] = Field()

    def __init__(
        self,
        position: Point3DLike = ORIGIN,
        color: ParsableManimColor = WHITE,
        intensity: float = DEFAULT_LIGHT_INTENSITY,
        **kwargs: Any,
    ) -> None:
        self.light_color = ManimColor(color)
        self.intensity = float(intensity)
        super().__init__(color=color, **kwargs)
        self.points = np.array([position], dtype=float)

    @property
    def position(self) -> Point3D:
        """Where the light is."""
        return self.points[0]

    def set_color(
        self,
        color: ParsableManimColor = WHITE,
        alpha: Any = None,
        family: bool = True,
    ) -> Self:
        self.light_color = ManimColor(color)
        return super().set_color(color, alpha, family)

    def get_color(self) -> ManimColor:
        return self.light_color

    def set_intensity(self, intensity: float) -> Self:
        self.intensity = float(intensity)
        return self

    def fade(self, darkness: float = 0.5, family: bool = True) -> Self:
        """Dim the light: :class:`.FadeOut` turns it off."""
        self.intensity = self.intensity * (1 - darkness)
        return self

    def interpolate_color(
        self, mobject1: Mobject, mobject2: Mobject, alpha: float
    ) -> Self:
        """Colour and intensity between two lights; the end values themselves at 1."""
        start, end = mobject1, mobject2
        if not (isinstance(start, PointLight) and isinstance(end, PointLight)):
            return self
        if alpha >= 1:
            self.light_color, self.intensity = end.light_color, end.intensity
        elif alpha <= 0:
            self.light_color, self.intensity = start.light_color, start.intensity
        else:
            self.light_color = start.light_color.interpolate(end.light_color, alpha)
            self.intensity = interpolate(start.intensity, end.intensity, alpha)
        return self
