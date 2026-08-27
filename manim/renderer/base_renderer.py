"""Structural interface shared by all Manim renderers.

Using ``typing.Protocol`` (rather than an ABC) means the existing
``CairoRenderer`` and ``OpenGLRenderer`` satisfy the interface automatically
without any inheritance changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

    from manim.mobject.mobject import Mobject
    from manim.mobject.value_tracker import ValueTracker
    from manim.scene.scene import Scene


# ---------------------------------------------------------------------------
# Camera protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ThreeDCameraProtocol(Protocol):
    """Interface that every 3D camera must satisfy.

    ``ThreeDScene`` calls these attributes and methods on
    ``renderer.camera``; declaring them here makes the contract explicit
    for all three renderers (Cairo ``ThreeDCamera``, ``OpenGLCamera``,
    and ``WebGPUCamera``).
    """

    # ── angle trackers (Cairo uses ValueTrackers; OpenGL/WebGPU store
    #    the angles directly and expose them as plain attributes) ─────────────
    theta_tracker: ValueTracker
    phi_tracker: ValueTracker
    gamma_tracker: ValueTracker
    focal_distance_tracker: ValueTracker
    zoom_tracker: ValueTracker

    # Mobject used as the camera frame-centre (moved to pan the scene).
    _frame_center: Mobject

    # ── orientation setters ──────────────────────────────────────────────────
    def set_phi(self, phi: float) -> None: ...
    def set_theta(self, theta: float) -> None: ...
    def set_gamma(self, gamma: float) -> None: ...
    def set_zoom(self, zoom: float) -> None: ...
    def set_focal_distance(self, focal_distance: float) -> None: ...

    # ── incremental rotation (OpenGL / WebGPU ambient rotation) ─────────────
    def increment_theta(self, dtheta: float) -> None: ...
    def increment_phi(self, dphi: float) -> None: ...
    def increment_gamma(self, dgamma: float) -> None: ...

    # ── updater support (camera is a Mobject in OpenGL / WebGPU) ────────────
    def add_updater(self, func: Any, **kwargs: Any) -> None: ...
    def clear_updaters(self) -> None: ...

    # ── moving-mobject tracking ──────────────────────────────────────────────
    def get_value_trackers(self) -> list[ValueTracker]: ...

    # ── fixed-orientation / fixed-in-frame helpers (Cairo) ──────────────────
    def add_fixed_orientation_mobjects(
        self, *mobjects: Mobject, **kwargs: Any
    ) -> None: ...
    def remove_fixed_orientation_mobjects(self, *mobjects: Mobject) -> None: ...
    def add_fixed_in_frame_mobjects(self, *mobjects: Mobject) -> None: ...
    def remove_fixed_in_frame_mobjects(self, *mobjects: Mobject) -> None: ...


# ---------------------------------------------------------------------------
# Renderer protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RendererProtocol(Protocol):
    """Protocol that every Manim renderer must satisfy.

    ``scene.py`` accesses these attributes and methods on the renderer object.
    Declaring them here makes the contract explicit and enables static type
    checking without breaking existing renderer classes.
    """

    # ── core attributes ──────────────────────────────────────────────────────
    camera: Any  # ThreeDCameraProtocol for 3D renderers; Any for 2D
    skip_animations: bool
    num_plays: int
    time: float
    file_writer: Any
    window: Any  # WebGPUWindow | pyglet window | None
    animation_start_time: float
    static_image: Any

    # Camera configuration dict (pixel_width, pixel_height, …).
    # Accessed by SpecialThreeDScene to choose low/high quality config.
    camera_config: dict

    # Set of currently-held key codes; polled by Scene.interact() and
    # the WebGPU window event handler.
    pressed_keys: set

    # ── lifecycle ────────────────────────────────────────────────────────────
    def init_scene(self, scene: Scene) -> None: ...

    def play(self, scene: Scene, *args: Any, **kwargs: Any) -> None: ...

    def render(
        self, scene: Scene, frame_offset: float, moving_mobjects: list
    ) -> None: ...

    def update_frame(self, scene: Scene) -> None: ...

    def scene_finished(self, scene: Scene) -> None: ...

    def clear_screen(self) -> None: ...

    # ── frame access ─────────────────────────────────────────────────────────
    def get_image(self) -> Image.Image: ...

    def get_frame(self) -> np.ndarray: ...
