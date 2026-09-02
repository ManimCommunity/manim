"""Point-cloud mobject for WebGPU TrueDot rendering.

``PointDot`` is a single dot rendered as a 3-D lit sphere.
``DotCloud3D`` is an N-point cloud rendered as N lit spheres.

These classes are ``Mobject``-based (not ``OpenGLMobject``-based) so they
work transparently with both Cairo scenes (skipped silently) and WebGPU
scenes (routed to the TrueDot pipeline).
"""

from __future__ import annotations

__all__ = ["DotCloud3D", "PointDot"]

from typing import Any

import numpy as np

from manim.constants import ORIGIN
from manim.mobject.mobject import Mobject
from manim.typing import Point3DLike
from manim.utils.color import WHITE, ParsableManimColor, color_to_rgba


class DotCloud3D(Mobject):
    """A cloud of points, each rendered as a lit sphere by the WebGPU renderer.

    In Cairo / OpenGL renderers, ``DotCloud3D`` objects are silently ignored
    (they produce no geometry for those pipelines).

    Parameters
    ----------
    points
        Array of world-space positions, shape (N, 3).
    color
        Base colour of all dots (can be overridden per-point via ``set_rgbas``).
    radius
        World-space radius of each sphere in scene units.
    gloss
        Specular shininess (Cairo-style): 0 = matte, 1 = very shiny.
    shadow
        Diffuse darkening strength: 0 = no shadow, 1 = full Lambert shading.
    """

    def __init__(
        self,
        points: np.ndarray | list | None = None,
        color: ParsableManimColor = WHITE,
        radius: float = 0.05,
        gloss: float = 0.3,
        shadow: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        pts = (
            np.zeros((0, 3), dtype=np.float32)
            if points is None
            else np.asarray(points, dtype=np.float32)
        )
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        self._cloud_points: np.ndarray = pts.astype(np.float32)
        self._rgbas: np.ndarray = np.tile(
            np.asarray(color_to_rgba(color), dtype=np.float32), (max(len(pts), 1), 1)
        )
        self.dot_radius: float = float(radius)
        self.gloss: float = float(gloss)
        self.shadow: float = float(shadow)
        # Set Mobject.points to the cloud positions so bounding-box helpers work.
        if len(pts) > 0:
            self.set_points(pts)

    # ------------------------------------------------------------------
    # Cloud-specific API
    # ------------------------------------------------------------------

    def get_cloud_points(self) -> np.ndarray:
        """Return the (N, 3) float32 array of dot centres."""
        return self._cloud_points

    def set_cloud_points(self, points: np.ndarray) -> DotCloud3D:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        self._cloud_points = pts
        if len(pts) > 0:
            self.set_points(pts)
        return self

    def get_rgbas(self) -> np.ndarray:
        """Return the (N, 4) float32 RGBA array for all dots."""
        return self._rgbas

    def set_rgbas(self, rgbas: np.ndarray) -> DotCloud3D:
        self._rgbas = np.asarray(rgbas, dtype=np.float32)
        return self

    def set_color(self, color: ParsableManimColor, family: bool = True) -> DotCloud3D:  # type: ignore[override]
        rgba = np.asarray(color_to_rgba(color), dtype=np.float32)
        self._rgbas = np.tile(rgba, (max(len(self._cloud_points), 1), 1))
        if family:
            for sub in self.submobjects:
                if isinstance(sub, DotCloud3D):
                    sub.set_color(color, family=False)
        return self

    def set_opacity(self, opacity: float, family: bool = True) -> DotCloud3D:  # type: ignore[override]
        self._rgbas[:, 3] = float(opacity)
        if family:
            for sub in self.submobjects:
                if isinstance(sub, DotCloud3D):
                    sub.set_opacity(opacity, family=False)
        return self

    # ------------------------------------------------------------------
    # Animation support — required Mobject overrides
    # ------------------------------------------------------------------

    def align_points_with_larger(self, larger_mobject: Mobject) -> None:
        """Tile _cloud_points and _rgbas to match the size of *larger_mobject*."""
        if not isinstance(larger_mobject, DotCloud3D):
            return
        n_target = len(larger_mobject._cloud_points)
        n_self = len(self._cloud_points)
        if n_self == 0 or n_self >= n_target:
            return
        reps = -(-n_target // n_self)  # ceiling division
        self._cloud_points = np.tile(self._cloud_points, (reps, 1))[:n_target]
        self._rgbas = np.tile(self._rgbas, (reps, 1))[:n_target]
        self.set_points(self._cloud_points)

    def interpolate_color(
        self, mobject1: Mobject, mobject2: Mobject, alpha: float
    ) -> None:
        """Linearly interpolate _rgbas between *mobject1* and *mobject2*."""
        if not isinstance(mobject1, DotCloud3D) or not isinstance(mobject2, DotCloud3D):
            return
        self._rgbas = ((1 - alpha) * mobject1._rgbas + alpha * mobject2._rgbas).astype(
            np.float32
        )

    def interpolate(
        self,
        mobject1: Mobject,
        mobject2: Mobject,
        alpha: float,
        path_func: Any = None,
    ) -> DotCloud3D:
        """Interpolate position and colour; keep _cloud_points in sync with points."""
        from manim.utils.bezier import interpolate as lerp

        if path_func is None:
            path_func = lerp
        super().interpolate(mobject1, mobject2, alpha, path_func)
        # Mobject.interpolate writes into self.points; mirror that into _cloud_points.
        self._cloud_points = np.asarray(self.points, dtype=np.float32)
        return self


class PointDot(DotCloud3D):
    """A single dot at *center* rendered as a lit sphere by the WebGPU renderer.

    Parameters
    ----------
    center
        World-space position of the dot.
    color
        Base colour.
    radius
        World-space radius of the sphere in scene units.
    gloss
        Specular shininess: 0 = matte, 1 = very shiny.
    shadow
        Diffuse darkening: 0 = flat, 1 = full Lambert shading.
    """

    def __init__(
        self,
        center: Point3DLike = ORIGIN,
        color: ParsableManimColor = WHITE,
        radius: float = 0.05,
        gloss: float = 0.3,
        shadow: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            points=np.asarray(center, dtype=np.float32).reshape(1, 3),
            color=color,
            radius=radius,
            gloss=gloss,
            shadow=shadow,
            **kwargs,
        )
