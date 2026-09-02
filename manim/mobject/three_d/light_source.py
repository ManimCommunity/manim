"""Light source mobjects for WebGPU 3-D rendering.

.. warning::

    **WebGPU renderer only.**  All classes in this module are silently ignored
    by the Cairo and OpenGL renderers.  They have no visual effect outside of
    scenes rendered with ``--renderer=webgpu``.

Classes
-------
LightSource
    Abstract base for all light types.  Extends :class:`~.Mobject` so it
    participates in scene management (``add``, ``remove``, ``play``).

AmbientLight
    Uniform omnidirectional light that brightens every surface equally.
    Only **one** ambient light may exist per scene; ``ThreeDScene`` adds a
    default one automatically.

DirectionalLight
    Parallel light from a fixed direction (like sunlight).  Intensity is
    constant regardless of position.

PointLight
    Omnidirectional light that radiates from a point in world space.  Falls
    off with the inverse-square of distance.

SpotLight
    Cone-shaped light from a point in a direction.  Same attenuation as
    ``PointLight`` but only illuminates within *cone_angle* of the direction.
    Soft penumbra can be controlled via the *penumbra* parameter.
"""

from __future__ import annotations

__all__ = ["AmbientLight", "DirectionalLight", "LightSource", "PointLight", "SpotLight"]

from typing import Any

import numpy as np

from manim.mobject.mobject import Mobject
from manim.typing import Point3DLike, Vector3D
from manim.utils.color import WHITE, ParsableManimColor, color_to_rgb

# ── Light kind constants (must match WGSL shader) ─────────────────────────────
_KIND_AMBIENT = 0
_KIND_DIRECTIONAL = 1
_KIND_POINT = 2
_KIND_SPOT = 3


class LightSource(Mobject):
    """Base class for all WebGPU light sources.

    Parameters
    ----------
    color
        Light colour.
    intensity
        Brightness scalar.  Typical range is [0, 1] for ambient/directional;
        higher values (e.g. 300) are suitable for point/spot lights with
        distance attenuation.
    **kwargs
        Forwarded to :class:`~.Mobject`.

    .. note::

        **WebGPU renderer only** — ignored by Cairo and OpenGL renderers.
    """

    # Subclasses must set this before calling super().__init__.
    _kind: int = -1

    def __init__(
        self,
        color: ParsableManimColor = WHITE,
        intensity: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.light_color: np.ndarray = np.asarray(color_to_rgb(color), dtype=np.float32)
        self.intensity: float = float(intensity)
        self.should_render: bool = False

    # ------------------------------------------------------------------
    # Packing helpers (used by the renderer)
    # ------------------------------------------------------------------

    def pack(self) -> bytes:
        """Return the 64-byte binary representation of this light.

        Matches the WGSL ``Light`` struct layout:

        .. code-block:: text

            offset  0  position   vec3<f32>  12 B
            offset 12  kind       u32         4 B
            offset 16  direction  vec3<f32>  12 B
            offset 28  intensity  f32         4 B
            offset 32  color      vec3<f32>  12 B
            offset 44  cone_angle f32         4 B
            offset 48  penumbra   f32         4 B
            offset 52  _pad0-2    f32×3      12 B  (alignment padding)
        """
        buf = np.zeros(16, dtype=np.float32)  # 16 × 4 B = 64 B
        buf[0:3] = self._get_position()
        buf[3] = np.float32(self._kind).view(np.float32)
        buf[4:7] = self._get_direction()
        buf[7] = self.intensity
        buf[8:11] = self.light_color
        buf[11] = self._get_cone_angle()
        buf[12] = self._get_penumbra()
        # buf[13], buf[14], buf[15] remain zero (padding)

        # Reinterpret index 3 as u32 so we get exact integer bit pattern.
        raw = buf.tobytes()
        kind_bytes = np.uint32(self._kind).tobytes()
        return raw[:12] + kind_bytes + raw[16:]

    # Subclass hooks — override as needed.
    def _get_position(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)

    def _get_direction(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)

    def _get_cone_angle(self) -> float:
        return 0.0

    def _get_penumbra(self) -> float:
        return 0.0


class AmbientLight(LightSource):
    """Uniform ambient light — illuminates every surface equally from all sides.

    Only **one** ambient light is allowed per scene.  ``ThreeDScene`` adds one
    by default (white, intensity 0.5).  Replacing it or adjusting its intensity
    gives global brightness control.

    Parameters
    ----------
    color
        Light colour.  Default: white.
    intensity
        Ambient brightness.  Default: ``0.5``.

    .. note::

        **WebGPU renderer only** — ignored by Cairo and OpenGL renderers.
    """

    _kind = _KIND_AMBIENT

    def __init__(
        self,
        color: ParsableManimColor = WHITE,
        intensity: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(color=color, intensity=intensity, **kwargs)


class DirectionalLight(LightSource):
    """Parallel directional light (like sunlight) — constant intensity everywhere.

    Parameters
    ----------
    direction
        World-space vector the light travels *toward* (points from light toward
        the scene).  Does not need to be normalised.  Default: ``[0, -1, -1]``
        (down-forward).
    color
        Light colour.  Default: white.
    intensity
        Brightness scalar.  Default: ``0.8``.

    .. note::

        **WebGPU renderer only** — ignored by Cairo and OpenGL renderers.
    """

    _kind = _KIND_DIRECTIONAL

    def __init__(
        self,
        direction: Vector3D = np.array([0.0, -1.0, -1.0]),
        color: ParsableManimColor = WHITE,
        intensity: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(color=color, intensity=intensity, **kwargs)
        d = np.asarray(direction, dtype=np.float32)
        norm = np.linalg.norm(d)
        self._direction: np.ndarray = (
            (d / norm) if norm > 1e-8 else np.array([0.0, 0.0, -1.0], dtype=np.float32)
        )

    def _get_direction(self) -> np.ndarray:
        return self._direction


class PointLight(LightSource):
    """Omnidirectional point light — radiates from a fixed world-space position.

    Intensity falls off with the inverse-square of the distance to the surface
    (``attenuation = intensity / dot(light_dir, light_dir)``).

    Parameters
    ----------
    position
        World-space centre of the light.  Default: ``[10, 10, -10]``.
    color
        Light colour.  Default: white.
    intensity
        Source brightness (before distance attenuation).  Default: ``300``.

    .. note::

        **WebGPU renderer only** — ignored by Cairo and OpenGL renderers.
    """

    _kind = _KIND_POINT

    def __init__(
        self,
        position: Point3DLike = np.array([10.0, 10.0, -10.0]),
        color: ParsableManimColor = WHITE,
        intensity: float = 300.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(color=color, intensity=intensity, **kwargs)
        self._position: np.ndarray = np.asarray(position, dtype=np.float32)

    def _get_position(self) -> np.ndarray:
        return self._position


class SpotLight(LightSource):
    """Cone-shaped point light.

    Like ``PointLight`` but only illuminates within *cone_angle* degrees of the
    *direction* vector.  A soft penumbra region of width *penumbra* degrees
    linearly fades the outer rim.

    Parameters
    ----------
    position
        World-space origin of the spot.  Default: ``[10, 10, -10]``.
    direction
        World-space vector the cone points toward.  Does not need to be
        normalised.  Default: ``[0, -1, -1]``.
    cone_angle
        Half-angle of the inner (full-brightness) cone, in **degrees**.
        Default: ``30``.
    penumbra
        Width of the soft penumbra region in **degrees**.  Default: ``5``.
    color
        Light colour.  Default: white.
    intensity
        Source brightness (before distance attenuation).  Default: ``300``.

    .. note::

        **WebGPU renderer only** — ignored by Cairo and OpenGL renderers.
    """

    _kind = _KIND_SPOT

    def __init__(
        self,
        position: Point3DLike = np.array([10.0, 10.0, -10.0]),
        direction: Vector3D = np.array([0.0, -1.0, -1.0]),
        cone_angle: float = 30.0,
        penumbra: float = 5.0,
        color: ParsableManimColor = WHITE,
        intensity: float = 300.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(color=color, intensity=intensity, **kwargs)
        self._position: np.ndarray = np.asarray(position, dtype=np.float32)
        d = np.asarray(direction, dtype=np.float32)
        norm = np.linalg.norm(d)
        self._direction: np.ndarray = (
            (d / norm) if norm > 1e-8 else np.array([0.0, 0.0, -1.0], dtype=np.float32)
        )
        self._cone_angle: float = float(cone_angle)
        self._penumbra: float = float(penumbra)

    def _get_position(self) -> np.ndarray:
        return self._position

    def _get_direction(self) -> np.ndarray:
        return self._direction

    def _get_cone_angle(self) -> float:
        return self._cone_angle

    def _get_penumbra(self) -> float:
        return self._penumbra
