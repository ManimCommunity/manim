"""WebGPU renderer for Manim — Phase 1.

Phase 1 scope
-------------
* Headless rendering (no preview window).
* VMobject fill only.
* ``config.save_last_frame = True`` → saves a PNG.
* ``config.write_to_movie = True`` → writes video frames.

Design
------
Reads geometry directly from Cairo ``VMobject``.  No dependency on OpenGL
classes (``OpenGLCamera``, ``OpenGLVMobject``, ``moderngl``).

Camera
------
A simple orthographic projection matrix maps Manim's frame coordinate system
(centre at origin, width = config.frame_width, height = config.frame_height)
to WebGPU NDC (x, y ∈ [-1, 1], z ∈ [0, 1]).
"""

from __future__ import annotations

import collections
import time
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from manim import config, logger
from manim.constants import OUT, PI, RIGHT
from manim.mobject.mobject import Mobject
from manim.mobject.three_d.light_source import LightSource
from manim.mobject.three_d.three_dimensions import Surface
from manim.mobject.types.image_mobject import (
    AbstractImageMobject,
    ImageMobjectFromCamera,
)
from manim.mobject.types.vectorized_mobject import VMobject
from manim.scene.scene_file_writer import SceneFileWriter
from manim.utils.color import color_to_rgba
from manim.utils.exceptions import EndSceneEarlyException
from manim.utils.hashing import get_hash_from_play_call
from manim.utils.iterables import list_update
from manim.utils.simple_functions import clip
from manim.utils.space_ops import (
    quaternion_from_angle_axis,
    quaternion_mult,
    rotation_matrix_transpose_from_quaternion,
)

from .webgpu_vmobject_rendering import (
    FILL_STROKE_VERTEX_LAYOUT,
    SURFACE_COMBINED_VERTEX_LAYOUT,
    TRUE_DOT_VERTEX_LAYOUT,
    DotCloud3D,
    build_true_dot_vbo,
    collect_frame_data,
    draw_frame_data,
    draw_frame_data_subcam,
)

if TYPE_CHECKING:
    import wgpu as wgpu_t

    from manim.scene.scene import Scene

    from .webgpu_renderer_window import WebGPUWindow

try:
    import wgpu
except ImportError as exc:
    msg = (
        "wgpu-py is required for the WebGPU renderer. "
        "Install it with:  pip install wgpu"
    )
    raise ImportError(msg) from exc


# ---------------------------------------------------------------------------
# Camera — feature-parity with OpenGLCamera for 2-D + 3-D scenes.
# ---------------------------------------------------------------------------


class WebGPUCamera(Mobject):
    """Camera for the WebGPU renderer.

    Inherits from ``Mobject`` so it can carry updaters and be added to the
    scene, matching the pattern used by ``OpenGLCamera(OpenGLMobject)``.

    Matches the attribute / method surface of ``OpenGLCamera`` so that
    scene code that inspects ``renderer.camera`` works without changes.

    Projection
    ----------
    * 2-D scenes: orthographic, z mapped to the WebGPU [0, 1] NDC
      range.
    * 3-D scenes: perspective projection driven by ``focal_distance``
      and the Euler-angle view matrix.

    Interactive controls (preview window only)
    ------------------------------------------
    When a preview window is open, :class:`WebGPUWindow` handles mouse
    events and mutates the camera in real time:

    * **Left-drag** — orbit: horizontal motion changes ``theta``; vertical
      motion changes ``phi`` (clamped to ``[minimum_polar_angle,
      maximum_polar_angle]``).
    * **Right-drag / middle-drag** — pan: translates the view laterally in
      camera space.  The pan offset is owned by the window and injected into
      ``view_matrix`` via ``_cam_pan_x`` / ``_cam_pan_y`` just before each
      render; it is not part of the scripted camera model.
    * **Scroll wheel** — zoom: adjusts ``focal_distance`` for perspective
      cameras or scales ``frame_shape`` for orthographic cameras.
    * **Key** ``r`` — reset: restores ``euler_angles`` and ``focal_distance``
      to their defaults and clears the window pan offset.

    Parameters
    ----------
    frame_shape
        (width, height) of the rendered frame.  Defaults to
        ``(config.frame_width, config.frame_height)``.
    frame_center
        World-space origin of the camera frame.  Defaults to the origin.
    euler_angles
        (theta, phi, gamma) camera orientation angles in radians.
        Defaults to (0, 0, 0) — looking straight down the −Z axis.
    focal_distance
        Perspective focal distance expressed as a multiple of ``frame_height``.
        Only used when ``orthographic=False``.
    orthographic
        Use orthographic (True) or perspective (False) projection.
        Default is True (matching Manim's default 2-D look).
    minimum_polar_angle / maximum_polar_angle
        Clamp range for the phi Euler angle during interactive orbit.
    """

    near: float = -100.0
    far: float = 100.0
    use_z_index: bool = True

    def __init__(
        self,
        frame_shape: tuple[float, float] | None = None,
        frame_center: np.ndarray | None = None,
        euler_angles: np.ndarray | None = None,
        focal_distance: float = 20.0,
        orthographic: bool = False,
        minimum_polar_angle: float = -PI / 2,
        maximum_polar_angle: float = PI / 2,
    ) -> None:
        super().__init__()
        self.use_z_index = True
        self.frame_rate: int = config.get("frame_rate", 60)
        self.orthographic = orthographic
        self.minimum_polar_angle = minimum_polar_angle
        self.maximum_polar_angle = maximum_polar_angle
        self.focal_distance = focal_distance

        self.frame_shape: tuple[float, float] = (
            frame_shape
            if frame_shape is not None
            else (float(config["frame_width"]), float(config["frame_height"]))
        )
        self.frame_center: np.ndarray = (
            np.asarray(frame_center, dtype=float)
            if frame_center is not None
            else np.array([0.0, 0.0, focal_distance], dtype=float)
        )
        # Default theta matches Cairo's default (-90°) so that the initial
        # rotation formula (theta + 90°) gives identity for 2-D scenes.
        self.euler_angles: np.ndarray = np.asarray(
            euler_angles if euler_angles is not None else [-PI / 2, 0.0, 0.0],
            dtype=float,
        )
        self.reset_rotation_matrix()

        # Fixed-mobject registries — populated by ThreeDScene helpers.
        # fixed_in_frame: objects rendered with identity rotation + ortho
        #   projection as a 2-D overlay on top of the 3-D scene (e.g. title text).
        # fixed_orientation: objects rendered with identity rotation + current
        #   projection so they don't tilt as the camera orbits (e.g. 3-D labels).
        self.fixed_in_frame_mobjects: set[Mobject] = set()
        self.fixed_orientation_mobjects: set[Mobject] = set()

        # ThreeDScene.get_moving_mobjects() checks _frame_center and
        # get_value_trackers() to detect camera-driven animation.
        # These are defined on ThreeDCamera (Cairo) but not on Mobject,
        # so we provide equivalent stubs here.
        self._frame_center: Mobject = Mobject()

        # ImageMobjectFromCamera registration — for ZoomedScene support.
        # Each frame the renderer renders the sub-camera view into a dedicated
        # GPU texture which is then composited as a regular image quad.
        self.image_mobjects_from_cameras: list = []

    def get_value_trackers(self) -> list:
        """Required by ThreeDScene.get_moving_mobjects.

        Returning ``[self]`` ensures that when the camera has updaters (e.g.
        ambient rotation), ThreeDScene.get_moving_mobjects() detects the camera
        in ``moving_mobjects`` and returns all scene mobjects — preventing the
        static-frame optimisation from freezing the 3-D scene under camera
        motion.
        """
        return [self]

    def get_mobjects_indicating_movement(self) -> list:
        """Return mobjects whose movement implies the whole scene is moving.

        Called by :class:`~.MovingCameraScene` to detect whether the camera
        frame (or any registered sub-camera frame) is animated, which forces
        all scene mobjects to be treated as moving so the static-frame
        optimisation is skipped.

        Mirrors ``MultiCamera.get_mobjects_indicating_movement`` so that
        :class:`~.ZoomedScene` works with the WebGPU renderer.
        """
        return [imfc.camera.frame for imfc in self.image_mobjects_from_cameras]

    # ------------------------------------------------------------------
    # Frame geometry helpers (mirrors OpenGLCamera)
    # ------------------------------------------------------------------

    def get_width(self) -> float:
        """Width of the camera frame in scene units."""
        return self.frame_shape[0]

    def get_height(self) -> float:
        """Height of the camera frame in scene units."""
        return self.frame_shape[1]

    def get_shape(self) -> tuple[float, float]:
        """(width, height) of the camera frame in scene units."""
        return self.frame_shape

    def get_center(self) -> np.ndarray:
        """World-space centre of the camera frame."""
        return self.frame_center.copy()

    def get_focal_distance(self) -> float:
        """Perspective focal distance in scene units."""
        return self.focal_distance * self.get_height()

    # ------------------------------------------------------------------
    # Camera reset
    # ------------------------------------------------------------------

    def to_default_state(self) -> WebGPUCamera:
        """Reset frame size, position, and orientation to config defaults."""
        self.frame_shape = (
            float(config["frame_width"]),
            float(config["frame_height"]),
        )
        self.frame_center = np.array([0.0, 0.0, self.focal_distance], dtype=float)
        self.euler_angles = np.array([-PI / 2, 0.0, 0.0])
        self.reset_rotation_matrix()
        return self

    # ------------------------------------------------------------------
    # Rotation — matches OpenGLCamera.set/increment_* interface
    # ------------------------------------------------------------------

    def reset_rotation_matrix(self) -> None:
        """Refresh the camera's inverse rotation matrix based on its Euler angles.

        The formula replicates Cairo's ThreeDCamera so that the same (theta, phi,
        gamma) values produce the same view in both renderers.

        Cairo's generate_rotation_matrix builds:
          R = R_z(gamma) @ R_x(-phi) @ R_z(-theta - 90°)   (np.dot loop order)
        and applies it to world column vectors in project_points.

        The WebGPU view matrix stores ``inverse_rotation_matrix`` and applies it
        directly.  ``rotation_matrix_transpose_from_quaternion(q)`` returns R_q^T
        where R_q is the rotation for quaternion q.  To get R_q^T == R_cairo we
        need:
          R_q = R_cairo^T = R_z(theta + 90°) @ R_x(phi) @ R_z(-gamma)
        i.e. the quaternion that rotates: first by -gamma around Z, then by phi
        around X, then by (theta + 90°) around Z:
          q = q(theta + PI/2, OUT) * q(phi, RIGHT) * q(-gamma, OUT)
        """
        theta, phi, gamma = self.euler_angles
        quat = quaternion_mult(
            quaternion_from_angle_axis(theta + PI / 2, OUT, axis_normalized=True),
            quaternion_from_angle_axis(phi, RIGHT, axis_normalized=True),
            quaternion_from_angle_axis(-gamma, OUT, axis_normalized=True),
        )
        self.inverse_rotation_matrix: np.ndarray = np.array(
            rotation_matrix_transpose_from_quaternion(np.asarray(quat, dtype=float)),
            dtype=float,
        )

    def set_euler_angles(
        self,
        theta: float | None = None,
        phi: float | None = None,
        gamma: float | None = None,
    ) -> WebGPUCamera:
        if theta is not None:
            self.euler_angles[0] = theta
        if phi is not None:
            self.euler_angles[1] = phi
        if gamma is not None:
            self.euler_angles[2] = gamma
        self.reset_rotation_matrix()
        return self

    def set_theta(self, theta: float) -> WebGPUCamera:
        return self.set_euler_angles(theta=theta)

    def set_phi(self, phi: float) -> WebGPUCamera:
        return self.set_euler_angles(phi=phi)

    def set_gamma(self, gamma: float) -> WebGPUCamera:
        return self.set_euler_angles(gamma=gamma)

    _PERSPECTIVE_FAR: float = 200.0

    def set_focal_distance(self, focal_distance: float) -> WebGPUCamera:
        """Set the perspective focal distance.

        Matches Cairo's ``ThreeDCamera.focal_distance`` convention: larger
        values push the camera further from the scene (same FOV, objects
        appear further away); smaller values pull it closer.

        The near plane is derived as ``focal_distance / 6`` so that the
        frustum height at depth ``focal_distance`` exactly equals the frame
        height — matching Cairo's perspective formula
        ``factor = focal_distance / (focal_distance - z_cam)``.

        ``focal_distance`` must be positive and less than ``_PERSPECTIVE_FAR``.
        Values outside that range are clamped.
        """
        max_fd = self._PERSPECTIVE_FAR * (1.0 - 1e-4)
        clamped = float(np.clip(focal_distance, 1e-4, max_fd))
        if clamped != focal_distance:
            logger.warning(
                "WebGPUCamera.set_focal_distance: value %.4g clamped to %.4g "
                "(must be in (0, far=%.4g))",
                focal_distance,
                clamped,
                self._PERSPECTIVE_FAR,
            )
        self.focal_distance = clamped
        # Keep the virtual camera position (frame_center z) in sync so that
        # the perspective projection exactly matches Cairo's formula at all depths.
        self.frame_center[2] = clamped
        return self

    def increment_theta(self, dtheta: float) -> WebGPUCamera:
        self.euler_angles[0] += dtheta
        self.reset_rotation_matrix()
        return self

    def increment_phi(self, dphi: float) -> WebGPUCamera:
        self.euler_angles[1] = clip(
            self.euler_angles[1] + dphi,
            self.minimum_polar_angle,
            self.maximum_polar_angle,
        )
        self.reset_rotation_matrix()
        return self

    def increment_gamma(self, dgamma: float) -> WebGPUCamera:
        self.euler_angles[2] += dgamma
        self.reset_rotation_matrix()
        return self

    # ------------------------------------------------------------------
    # View matrix (world → camera space)
    # ------------------------------------------------------------------

    @property
    def view_matrix(self) -> np.ndarray:
        """4×4 float32 view matrix: rotates and translates world space into
        camera space.

        Uses T(-c) @ R_inv, which rotates the world around the origin (matches
        OpenGLCamera behavior where the camera orbits the focal point).

        ``_cam_pan_x`` / ``_cam_pan_y`` are injected by :class:`WebGPUWindow`
        just before each render call.  They are not initialised in ``__init__``
        so that the camera model stays free of window/interaction state.
        ``getattr`` defaults to 0 when no window is attached.
        """
        R = np.asarray(self.inverse_rotation_matrix, dtype=np.float32)  # 3×3
        c = self.frame_center.astype(np.float32)
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = R
        # Camera-space translation: orbit distance along -Z plus lateral pan.
        # Positive pan_x shifts the scene right (camera moves left), matching
        # the "drag scene to the right" expectation for right-drag pan.
        pan_x = float(getattr(self, "_cam_pan_x", 0.0))
        pan_y = float(getattr(self, "_cam_pan_y", 0.0))
        view[:3, 3] = [-pan_x, -pan_y, -c[2]]
        return view

    @property
    def fixed_view_matrix(self) -> np.ndarray:
        """View matrix with camera rotation stripped — z-translation only.

        Used for fixed-orientation and fixed-in-frame mobjects so they don't
        tilt or spin when the camera orbits.  The z-translation is preserved so
        depth ordering within the fixed layer is consistent with the main scene.
        """
        view = np.eye(4, dtype=np.float32)
        view[2, 3] = -float(self.frame_center[2])
        return view

    @property
    def ortho_projection_matrix(self) -> np.ndarray:
        """Forced orthographic projection matrix, regardless of self.orthographic.

        Fixed-in-frame overlays always use orthographic so that screen-space
        coordinates map directly to Manim scene units (matching 2-D scenes).
        """
        # Orthographic: map frame to NDC with z ∈ [0, 1].
        # Note: Manim's +Z is out of the screen (towards the viewer).
        # So +Z should map to 0 (near) and -Z should map to 1 (far).
        # Z_clip = -1/(far-near) * Z + far/(far-near)
        fw, fh = self.frame_shape
        near, far = self.near, self.far
        return np.array(
            [
                [2.0 / fw, 0.0, 0.0, 0.0],
                [0.0, 2.0 / fh, 0.0, 0.0],
                [0.0, 0.0, -1.0 / (far - near), far / (far - near)],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Projection matrix (used by the shader uniform upload)
    # ------------------------------------------------------------------

    @property
    def projection_matrix(self) -> np.ndarray:
        """4×4 float32 projection matrix in WebGPU NDC convention (z ∈ [0, 1]).

        Perspective when ``self.orthographic`` is False (default).

        Design: the near plane is derived as ``focal_distance / 6`` so that the
        visible height at camera depth ``focal_distance`` (where world_z = 0
        maps to) exactly equals the frame height.  Combined with
        ``frame_center_z = focal_distance``, this exactly replicates Cairo's
        perspective formula ``factor = focal_distance / (focal_distance - z_cam)``
        (for all depths, not just at world_z = 0).
        """
        fw, fh = self.frame_shape

        if self.orthographic:
            return self.ortho_projection_matrix
        else:
            # n = fd/6, w = fw/6, h = fh/6  →  2n/w = 2*fd/fw, 2n/h = 2*fd/fh
            # → NDC_y = (2*fd/fh) * y / (-z_view)
            #         = (2*fd/fh) * y / (fd - z_cairo)  [with z_view = z_cairo - fd]
            # Cairo: NDC_y = fd/(fd-z_cairo) * y / (fh/2) = 2*fd/fh * y / (fd-z_cairo)  ✓
            f = self._PERSPECTIVE_FAR
            fd = self.focal_distance
            n = float(np.clip(fd / 6.0, 1e-6, f * (1.0 - 1e-4)))
            w, h = fw / 6.0, fh / 6.0
            return np.array(
                [
                    [2.0 * n / w, 0.0, 0.0, 0.0],
                    [0.0, 2.0 * n / h, 0.0, 0.0],
                    [0.0, 0.0, f / (n - f), n * f / (n - f)],
                    [0.0, 0.0, -1.0, 0.0],
                ],
                dtype=np.float32,
            )

    # ------------------------------------------------------------------
    # Fixed-mobject registry (used by ThreeDScene)
    # ------------------------------------------------------------------

    def add_fixed_in_frame_mobjects(self, *mobjects: Mobject) -> None:
        """Register mobjects to be rendered as 2-D screen-space overlays.

        These objects are drawn after the 3-D scene with a fresh depth buffer,
        identity camera rotation, and an orthographic projection so they always
        appear on top at their 2-D screen-space coordinates.
        """
        self.fixed_in_frame_mobjects.update(mobjects)

    def remove_fixed_in_frame_mobjects(self, *mobjects: Mobject) -> None:
        """Unregister mobjects previously added with add_fixed_in_frame_mobjects."""
        self.fixed_in_frame_mobjects.difference_update(mobjects)

    def add_fixed_orientation_mobjects(self, *mobjects: Mobject) -> None:
        """Register mobjects whose orientation is frozen relative to the camera.

        These objects still move in 3-D space (their world coordinates are used
        normally) but the camera rotation is not applied — they remain upright as
        the camera orbits.  Useful for 3-D labels that should always face forward.
        """
        self.fixed_orientation_mobjects.update(mobjects)

    def remove_fixed_orientation_mobjects(self, *mobjects: Mobject) -> None:
        """Unregister mobjects previously added with add_fixed_orientation_mobjects."""
        self.fixed_orientation_mobjects.difference_update(mobjects)

    def add_image_mobject_from_camera(self, image_mob_from_camera: Any) -> None:
        """Register an ImageMobjectFromCamera for sub-camera rendering.

        Called by :class:`~.ZoomedScene` when zooming is activated.  Each
        registered mob is rendered from its associated ``MovingCamera``'s
        perspective into a dedicated GPU texture every frame.

        **WebGPU renderer only** — this method is a no-op for Cairo / OpenGL.
        """
        if image_mob_from_camera not in self.image_mobjects_from_cameras:
            self.image_mobjects_from_cameras.append(image_mob_from_camera)

    def remove_image_mobject_from_camera(self, image_mob_from_camera: Any) -> None:
        """Unregister an ImageMobjectFromCamera previously added via
        ``add_image_mobject_from_camera``.
        """
        if image_mob_from_camera in self.image_mobjects_from_cameras:
            self.image_mobjects_from_cameras.remove(image_mob_from_camera)


# ---------------------------------------------------------------------------
# Main renderer class
# ---------------------------------------------------------------------------


class WebGPURenderer:
    """WebGPU renderer for Manim — headless and interactive preview.

    Supports PNG / video output in headless mode and a live preview window
    (enabled by ``config.preview = True`` or the ``-p`` CLI flag).

    Preview window controls
    -----------------------
    When the preview window is open, the camera can be navigated interactively
    with the mouse.  All interactions take effect immediately without restarting
    the scene.

    ========================  ================================================
    Input                     Action
    ========================  ================================================
    Left-drag                 **Orbit** — horizontal drag rotates theta (yaw);
                              vertical drag tilts phi (pitch), clamped to ±90°.
    Right-drag / Middle-drag  **Pan** — translates the view laterally in camera
                              space, proportional to the current frame size.
    Scroll wheel              **Zoom** — perspective: adjusts ``focal_distance``
                              exponentially (~12 % per notch); orthographic:
                              scales ``frame_shape`` by the same factor.
    Key ``r``                 **Reset** — restores the camera's default orbit,
                              zoom, and clears any accumulated pan offset.
    Key ``q``                 **Quit** — closes the preview window.
    ========================  ================================================

    Customising window interaction
    ------------------------------
    Subclass :class:`~.WebGPUWindow` and pass it via ``window_class`` to
    override any interaction hook without modifying the renderer itself:

    ========================  ================================================
    Hook to override          Triggered by
    ========================  ================================================
    ``on_mouse_drag``         Pointer move while a button is held.
    ``on_scroll``             Wheel / scroll event.
    ``on_key_press``          Key pressed down.
    ``on_key_release``        Key released.
    ``on_mouse_left_click``   Left button clicked (no drag).
    ``on_mouse_right_click``  Right button clicked (no drag).
    ========================  ================================================

    Building-block helpers ``orbit(dx, dy)``, ``pan(dx, dy)``, and
    ``zoom(scroll_dy)`` are also overridable for finer control.  See
    :class:`~.WebGPUWindow` for a full example.

    MSAA
    ----
    Pass ``msaa_samples=4`` to enable 4× multisample anti-aliasing.  This
    smooths geometric edges on surfaces, images, and dot-clouds (VMobjects
    already use SDF/coverage-based AA so the gain for pure-2D scenes is
    modest).  MSAA bypasses the static-frame optimisation, re-rendering every
    frame in full.
    """

    def __init__(
        self,
        file_writer_class: type[SceneFileWriter] = SceneFileWriter,
        skip_animations: bool = False,
        msaa_samples: int = 1,
        window_class: type | None = None,
    ) -> None:
        """Create a WebGPU renderer.

        Parameters
        ----------
        file_writer_class:
            Class used to write frames to disk.
        skip_animations:
            When True the renderer skips all animations (used for caching).
        msaa_samples:
            Multisample Anti-Aliasing sample count.  Must be 1 (disabled) or 4.
            MSAA smooths geometric edges of surfaces, images, and dot-clouds.
            VMobjects already use SDF/coverage-based AA, so the visual gain for
            pure-2D scenes is modest; the benefit is most visible for 3-D
            scenes with surface meshes and ``DotCloud3D``.

            When MSAA is enabled the static-frame optimisation is bypassed
            (every frame is fully re-rendered), which increases per-frame GPU
            work.  This is acceptable because MSAA already implies a quality-
            over-speed trade-off.
        window_class:
            Class used to create the preview window.  Must be
            :class:`~.WebGPUWindow` or a subclass of it.  Defaults to
            :class:`~.WebGPUWindow`.  Pass a subclass to customise mouse/
            keyboard interaction by overriding :meth:`~.WebGPUWindow.on_mouse_drag`,
            :meth:`~.WebGPUWindow.on_scroll`, :meth:`~.WebGPUWindow.on_key_press`,
            or :meth:`~.WebGPUWindow.on_key_release`.
        """
        if msaa_samples not in (1, 4):
            msg = f"msaa_samples must be 1 or 4, got {msaa_samples}"
            raise ValueError(msg)
        self._msaa_samples = msaa_samples
        self._window_class = window_class
        self._file_writer_class = file_writer_class
        self._original_skipping_status = skip_animations
        self.skip_animations = skip_animations

        self.animation_start_time: float = 0.0
        self.animation_elapsed_time: float = 0.0
        self.time: float = 0.0
        self.num_plays: int = 0
        self.animations_hashes: list[str | None] = []

        self.camera: WebGPUCamera = WebGPUCamera()
        self.window: WebGPUWindow | None = None
        self.pressed_keys: set[int] = set()
        self._static_image: Any = None
        self.file_writer: SceneFileWriter | None = None  # set by init_scene()

        # Static-frame compositing (WP1).
        # save_static_frame_data() renders static mobjects once into
        # _static_texture; update_frame() blits it as the background and only
        # re-draws the moving subset each animation frame.
        self._static_texture: wgpu_t.GPUTexture | None = None
        self._static_texture_view: wgpu_t.GPUTextureView | None = None
        self._has_static_frame: bool = False
        # IDs (id()) of the top-level mobjects that belong to the static layer.
        # Used in render() to partition scene.mobjects into static vs dynamic.
        self._static_mob_ids: set[int] = set()

        # SpecialThreeDScene reads renderer.camera_config["pixel_width"] to decide
        # whether to apply low-quality overrides.  Mirrors the pattern used by
        # OpenGLRenderer so that SpecialThreeDScene works unchanged with WebGPU.
        self.camera_config: dict = {
            "pixel_width": config.pixel_width,
            "pixel_height": config.pixel_height,
        }

        self.background_color = config["background_color"]

        # Filled by init_scene():
        self._device: wgpu_t.GPUDevice | None = None
        self._render_texture: wgpu_t.GPUTexture | None = None
        self._render_texture_view: wgpu_t.GPUTextureView | None = None
        self._depth_texture: wgpu_t.GPUTexture | None = None
        self._depth_texture_view: wgpu_t.GPUTextureView | None = None
        self._proj_bgl: wgpu_t.GPUBindGroupLayout | None = None

        # MSAA textures — created only when msaa_samples > 1.
        # _msaa_texture:       bgra8unorm, sample_count=N, RENDER_ATTACHMENT only.
        #                      Used as the draw target in Pass 1; resolves to
        #                      _render_texture at the end of each pass.
        # _msaa_depth_texture: depth24plus, sample_count=N, RENDER_ATTACHMENT only.
        #                      Must match the sample count of all MSAA pipelines.
        self._msaa_texture: wgpu_t.GPUTexture | None = None
        self._msaa_texture_view: wgpu_t.GPUTextureView | None = None
        self._msaa_depth_texture: wgpu_t.GPUTexture | None = None
        self._msaa_depth_texture_view: wgpu_t.GPUTextureView | None = None

        # Combined fill+stroke pipelines (vmobject_fill_stroke.wgsl).
        # _fill_stroke_bgl is reused for both compute output and render input
        # (camera uniform + read-only quads storage).
        self._fill_stroke_bgl: wgpu_t.GPUBindGroupLayout | None = None
        # Main pipelines — multisample count matches self._msaa_samples.
        # Used in Pass 1 (the MSAA main-render pass).
        self._fill_stroke_pipeline: wgpu_t.GPURenderPipeline | None = (
            None  # 2-D, no depth write
        )
        self._fill_stroke_3d_pipeline: wgpu_t.GPURenderPipeline | None = (
            None  # 3-D, depth write
        )
        # Overlay pipelines — always count=1.
        # Used in Pass 4 (fixed-in-frame overlay, renders to _render_texture_view
        # at sample_count=1 after the MSAA resolve has already completed).
        self._fill_stroke_pipeline_1x: wgpu_t.GPURenderPipeline | None = None
        self._fill_stroke_3d_pipeline_1x: wgpu_t.GPURenderPipeline | None = None

        # Compute pipeline: cubic_to_quads.wgsl.
        self._compute_bgl: wgpu_t.GPUBindGroupLayout | None = None
        self._cubic_to_quads_pipeline: wgpu_t.GPUComputePipeline | None = None

        # Surface pipelines: opaque (depth write, combined fill+wireframe) and OIT.
        self._surface_pipeline: wgpu_t.GPURenderPipeline | None = None  # opaque
        self._surface_oit_pipeline: wgpu_t.GPURenderPipeline | None = None
        # OIT accumulation textures (rgba16float each).
        self._oit_accum_texture: wgpu_t.GPUTexture | None = None
        self._oit_accum_view: wgpu_t.GPUTextureView | None = None
        self._oit_reveal_texture: wgpu_t.GPUTexture | None = None
        self._oit_reveal_view: wgpu_t.GPUTextureView | None = None
        # OIT composition pipeline + bind group.
        self._oit_compose_pipeline: wgpu_t.GPURenderPipeline | None = None
        self._oit_compose_bgl: wgpu_t.GPUBindGroupLayout | None = None
        self._oit_compose_bind_group: wgpu_t.GPUBindGroup | None = None

        # TrueDot pipeline (true_dot.wgsl) — renders DotCloud3D/PointDot as
        # screen-aligned lit sphere quads (CPU-expanded, 6 verts per dot).
        self._true_dot_pipeline: wgpu_t.GPURenderPipeline | None = None

        # Sub-camera pipelines — identical to the main pipelines but target
        # rgba8unorm instead of bgra8unorm so the rendered texture can be
        # sampled by the image pipeline without B↔R channel confusion.
        # Used for ImageMobjectFromCamera (ZoomedScene support).
        self._sub_cam_fill_stroke_pipeline: wgpu_t.GPURenderPipeline | None = None
        self._sub_cam_fill_stroke_3d_pipeline: wgpu_t.GPURenderPipeline | None = None
        self._sub_cam_surface_pipeline: wgpu_t.GPURenderPipeline | None = None
        # Per-mob GPU resource cache: mob id → dict with render texture,
        # depth texture, uniform buffer, camera bind group, tex bind group.
        self._sub_cam_resources: dict[int, dict] = {}

        # Image pipeline (image.wgsl) — renders ImageMobject pixel arrays as
        # textured quads before the VMobject pass (painter's algorithm).
        self._image_pipeline: wgpu_t.GPURenderPipeline | None = None
        self._image_tex_bgl: wgpu_t.GPUBindGroupLayout | None = None
        self._image_tint_bgl: wgpu_t.GPUBindGroupLayout | None = None
        # Cache: ImageMobject → (fingerprint, GPUTexture, GPUBindGroup).
        # Keyed weakly so destroyed mobs release their GPU textures.
        self._image_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        # Cache: ImageMobject → (points_fingerprint, GPUBuffer).
        # VBO is reused across frames as long as mob.points[:4] hasn't changed.
        self._image_vbo_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        # Cache: ImageMobject → (tint_fingerprint, GPUBuffer, GPUBindGroup).
        # Tint bind group is rebuilt only when mob.color changes.
        self._image_tint_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

        # Compact readback compute pipeline (GPU row-depadding + B↔R fix).
        self._readback_compute_pipeline: wgpu_t.GPUComputePipeline | None = None
        self._readback_compute_bgl: wgpu_t.GPUBindGroupLayout | None = None
        self._readback_compute_bind_group: wgpu_t.GPUBindGroup | None = None
        # Storage buffer the compute shader writes into (STORAGE | COPY_SRC).
        # One shared buffer — the compute shader always writes here first.
        self._readback_storage_buf: wgpu_t.GPUBuffer | None = None

        # ── Staging-buffer pool ──────────────────────────────────────────
        # Instead of a single MAP_READ buffer we keep a ring of N buffers.
        # update_frame() writes the readback into the next free slot and
        # starts map_async immediately.  _get_mapped_frame_array() dequeues
        # the oldest in-flight slot; by the time N frames have been submitted
        # the GPU has had N frame-times to finish and sync_wait() returns
        # instantly with no pipeline stall.
        _READBACK_POOL = 3  # triple-buffering
        self._READBACK_POOL: int = _READBACK_POOL
        self._readback_pool: list[Any] = []  # GPUBuffer × _READBACK_POOL
        # FIFO of slot indices submitted but not yet read.
        self._readback_queue: collections.deque = collections.deque()
        # Ring write pointer: next pool slot for update_frame to fill.
        self._readback_write_slot: int = 0

        # Cache the last successfully read pixel array.  Cleared by
        # update_frame() so that get_image() / get_frame() hit the GPU path
        # only once per rendered frame; repeated calls within the same frame
        # are served from this cache without any GPU interaction.
        self._readback_cache: np.ndarray | None = None

        # Per-frame state (set during update_frame, cleared after submit).
        self.current_render_pass: wgpu_t.GPURenderPassEncoder | None = None
        self.camera_bind_group: wgpu_t.GPUBindGroup | None = None
        self._camera_uniform_buf: wgpu_t.GPUBuffer | None = None
        # Fixed-mobject bind groups (rebuilt each frame with stripped-rotation view).
        #   fixed_camera_bind_group: identity rotation + current projection (fixed-orientation)
        #   fixed_frame_bind_group:  identity rotation + orthographic projection (fixed-in-frame)
        self.fixed_camera_bind_group: wgpu_t.GPUBindGroup | None = None
        self._fixed_orient_uniform_buf: wgpu_t.GPUBuffer | None = None
        self.fixed_frame_bind_group: wgpu_t.GPUBindGroup | None = None
        self._fixed_frame_uniform_buf: wgpu_t.GPUBuffer | None = None
        self.frame_vbos: list[wgpu_t.GPUBuffer] = []

        # _FrameData cache: keyed by cache slot name ("normal", "orient", "frame").
        # Each entry is (fingerprint_bytes, _FrameData).  On a fingerprint hit we
        # return the cached _FrameData, skipping all tessellation and buffer uploads.
        self._fd_cache: dict[str, tuple[bytes, Any]] = {}

    # ------------------------------------------------------------------
    # static_image property — scene.py sets this to None at end of play()
    # ------------------------------------------------------------------

    @property
    def static_image(self) -> Any:
        return self._static_image

    @static_image.setter
    def static_image(self, value: Any) -> None:
        self._static_image = value
        if value is None:
            self._has_static_frame = False
            self._static_mob_ids = set()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_scene(self, scene: Scene) -> None:
        """Create the wgpu device, offscreen texture, and file writer."""
        self.scene = scene

        if self._device is not None:
            # Rerun path — reuse the existing GPU device, textures, and window.
            # Only reset the per-scene state so the renderer is ready for a
            # fresh construct() call without tearing down and recreating GPU
            # resources (which would lose the live preview window).
            self.partial_movie_files = []
            self.file_writer = self._file_writer_class(self, scene.__class__.__name__)
            self.background_color = config["background_color"]
            return

        self.partial_movie_files: list[str | None] = []
        self.file_writer: SceneFileWriter = self._file_writer_class(
            self,
            scene.__class__.__name__,
        )

        self.background_color = config["background_color"]

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self._device = adapter.request_device_sync(
            required_features=[],
            required_limits={},
        )
        logger.debug("WebGPU adapter: %s", adapter.info)

        width = config.pixel_width
        height = config.pixel_height
        # bgra8unorm matches the window surface format on all major platforms
        # (Metal/Vulkan/DX12), enabling copy_texture_to_texture without a
        # blit shader.  Readback in _get_mapped_frame_array() swaps B↔R to
        # produce the RGBA output expected by PIL / numpy callers.
        self._render_texture = self._device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.bgra8unorm,
            usage=(
                wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.COPY_DST  # receives blit from _static_texture
                | wgpu.TextureUsage.TEXTURE_BINDING  # read by compact-readback compute shader
            ),
        )
        self._render_texture_view = self._render_texture.create_view()

        # Static-frame texture: stores the pre-rendered static layer.
        # Populated once per animation by save_static_frame_data(); blitted
        # back into _render_texture each frame by update_frame(blit_static=True).
        self._static_texture = self._device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.bgra8unorm,
            usage=(
                wgpu.TextureUsage.COPY_DST  # written by copy from _render_texture
                | wgpu.TextureUsage.COPY_SRC  # read back into _render_texture each frame
            ),
        )
        self._static_texture_view = self._static_texture.create_view()

        self._depth_texture = self._device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.depth24plus,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        self._depth_texture_view = self._depth_texture.create_view()

        # MSAA textures — only when msaa_samples > 1.
        if self._msaa_samples > 1:
            self._msaa_texture = self._device.create_texture(
                size=(width, height, 1),
                format=wgpu.TextureFormat.bgra8unorm,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
                sample_count=self._msaa_samples,
            )
            self._msaa_texture_view = self._msaa_texture.create_view()
            self._msaa_depth_texture = self._device.create_texture(
                size=(width, height, 1),
                format=wgpu.TextureFormat.depth24plus,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
                sample_count=self._msaa_samples,
            )
            self._msaa_depth_texture_view = self._msaa_depth_texture.create_view()

        self._proj_bgl = self._create_camera_bgl()

        # Surface mesh lines sit exactly on the surface triangles.  A negative
        # depth bias pulls each fragment slightly toward the camera so the mesh
        # always wins the depth test without visually offsetting the lines.
        # depth_bias=-10000 gives ~6e-4 constant offset in [0,1] depth space
        # (depth24plus unit ≈ 6e-8), which is large enough to reliably beat
        # floating-point depth jitter on flat/low-slope surface regions where
        # depth_bias_slope_scale alone contributes nearly zero.
        self._surface_pipeline = self._create_surface_pipeline(
            self._proj_bgl,
            cull_mode="none",
            depth_write=True,
            msaa_samples=self._msaa_samples,
        )

        # Combined fill+stroke pipeline (replaces separate slug + stroke pipelines).
        self._fill_stroke_bgl, self._fill_stroke_pipeline = (
            self._create_fill_stroke_pipeline(
                depth_test=False, msaa_samples=self._msaa_samples
            )
        )
        _, self._fill_stroke_3d_pipeline = self._create_fill_stroke_pipeline(
            depth_test=True, msaa_samples=self._msaa_samples
        )

        # Overlay pipelines — always count=1.  Used in Pass 4 (fixed-in-frame)
        # which renders directly into _render_texture_view after the MSAA resolve.
        if self._msaa_samples > 1:
            _, self._fill_stroke_pipeline_1x = self._create_fill_stroke_pipeline(
                depth_test=False, msaa_samples=1
            )
            _, self._fill_stroke_3d_pipeline_1x = self._create_fill_stroke_pipeline(
                depth_test=True, msaa_samples=1
            )
        else:
            # When MSAA is off the overlay pipelines are the same objects.
            self._fill_stroke_pipeline_1x = self._fill_stroke_pipeline
            self._fill_stroke_3d_pipeline_1x = self._fill_stroke_3d_pipeline

        # GPU compute: cubic → quadratic conversion.
        self._compute_bgl, self._cubic_to_quads_pipeline = (
            self._create_cubic_to_quads_pipeline()
        )

        self._create_oit_resources(width, height)
        self._create_readback_pipeline(width, height)
        self._image_tex_bgl, self._image_tint_bgl, self._image_pipeline = (
            self._create_image_pipeline(msaa_samples=self._msaa_samples)
        )
        self._true_dot_pipeline = self._create_true_dot_pipeline(
            self._proj_bgl,
            msaa_samples=self._msaa_samples,
        )

        # Sub-camera pipelines (rgba8unorm target) for ZoomedScene support.
        # Sub-camera targets are always count=1 (they render into their own
        # rgba8unorm textures, not into the MSAA main buffer).
        _, self._sub_cam_fill_stroke_pipeline = self._create_fill_stroke_pipeline(
            depth_test=False, target_format="rgba8unorm", msaa_samples=1
        )
        _, self._sub_cam_fill_stroke_3d_pipeline = self._create_fill_stroke_pipeline(
            depth_test=True, target_format="rgba8unorm", msaa_samples=1
        )
        self._sub_cam_surface_pipeline = self._create_surface_pipeline(
            self._proj_bgl,
            cull_mode="none",
            depth_write=True,
            target_format="rgba8unorm",
            msaa_samples=1,
        )

        # Persistent camera uniform buffers — created once, updated each frame via
        # write_buffer.  Using COPY_DST so queue.write_buffer can write into them.
        # These stable GPU objects let cached _FrameData bind groups remain valid
        # across frames: the bind group references the same buffer; write_buffer
        # updates its contents so the shader always sees the current camera.
        # Uniform buffer size: proj(64) + view(64) + num_lights+pad(16) + Light×8(512) = 656 B
        _UBO_SIZE = 656
        self._camera_uniform_buf = self._device.create_buffer(
            size=_UBO_SIZE,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._fixed_orient_uniform_buf = self._device.create_buffer(
            size=_UBO_SIZE,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._fixed_frame_uniform_buf = self._device.create_buffer(
            size=_UBO_SIZE,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # Persistent camera bind groups — constant layout + constant buffer objects,
        # so they never need to be recreated.
        def _make_persistent_bg(buf: wgpu_t.GPUBuffer) -> wgpu_t.GPUBindGroup:
            return self._device.create_bind_group(
                layout=self._proj_bgl,
                entries=[
                    {
                        "binding": 0,
                        "resource": {"buffer": buf, "offset": 0, "size": _UBO_SIZE},
                    }
                ],
            )

        self.camera_bind_group = _make_persistent_bg(self._camera_uniform_buf)
        self.fixed_camera_bind_group = _make_persistent_bg(
            self._fixed_orient_uniform_buf
        )
        self.fixed_frame_bind_group = _make_persistent_bg(self._fixed_frame_uniform_buf)

        if self.should_create_window():
            from .webgpu_renderer_window import WebGPUWindow

            wclass = self._window_class or WebGPUWindow
            self.window = wclass(self)

    # ------------------------------------------------------------------
    # Pipeline creation
    # ------------------------------------------------------------------

    def _create_camera_bgl(self) -> wgpu_t.GPUBindGroupLayout:
        """Create the bind group layout shared by stroke, surface, and Slug pipelines.

        Layout: binding 0 — one uniform buffer (656 bytes total):
          offset   0 — projection  mat4x4<f32>  64 B
          offset  64 — view        mat4x4<f32>  64 B
          offset 128 — num_lights  u32           4 B
          offset 132 — _pad        u32 × 3      12 B
          offset 144 — lights      Light × 8   512 B  (each Light = 64 B)
        """
        assert self._device is not None
        return self._device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                }
            ]
        )

    def _create_fill_stroke_pipeline(
        self,
        depth_test: bool = False,
        target_format: str = "bgra8unorm",
        msaa_samples: int = 1,
    ) -> tuple[wgpu_t.GPUBindGroupLayout, wgpu_t.GPURenderPipeline]:
        """Create the combined fill+stroke pipeline (vmobject_fill_stroke.wgsl).

        The bind group layout mirrors the slug fill layout:
          binding 0 — camera uniform (656 bytes)
          binding 1 — quads storage buffer (read-only, output of compute shader)

        depth_test=False — 2-D objects: depth-read-only (painter's algorithm).
        depth_test=True  — 3-D objects: depth-write + depth-test.
        """
        assert self._device is not None
        shader_path = Path(__file__).parent / "shaders" / "vmobject_fill_stroke.wgsl"
        shader_module = self._device.create_shader_module(
            code=shader_path.read_text(encoding="utf-8")
        )

        bgl = self._device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {
                        "type": "read-only-storage",
                        "has_dynamic_offset": False,
                    },
                },
            ]
        )

        _blend = {
            "color": {
                "src_factor": "src-alpha",
                "dst_factor": "one-minus-src-alpha",
                "operation": "add",
            },
            "alpha": {
                "src_factor": "one",
                "dst_factor": "one",
                "operation": "add",
            },
        }

        pipeline = self._device.create_render_pipeline(
            layout=self._device.create_pipeline_layout(bind_group_layouts=[bgl]),
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [FILL_STROKE_VERTEX_LAYOUT],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": getattr(wgpu.TextureFormat, target_format),
                        "blend": _blend,
                    }
                ],
            },
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": depth_test,
                "depth_compare": "less",
                "stencil_front": {
                    "compare": "always",
                    "fail_op": "keep",
                    "depth_fail_op": "keep",
                    "pass_op": "keep",
                },
                "stencil_back": {
                    "compare": "always",
                    "fail_op": "keep",
                    "depth_fail_op": "keep",
                    "pass_op": "keep",
                },
                "stencil_read_mask": 0,
                "stencil_write_mask": 0,
            },
            multisample={
                "count": msaa_samples,
                "mask": 0xFFFF_FFFF,
                "alpha_to_coverage_enabled": False,
            },
        )
        return bgl, pipeline

    def _create_cubic_to_quads_pipeline(
        self,
    ) -> tuple[wgpu_t.GPUBindGroupLayout, wgpu_t.GPUComputePipeline]:
        """Create the compute pipeline that converts cubics → quadratics.

        Bind group layout:
          binding 0 — input  cubics  (read-only-storage, 12 floats/cubic)
          binding 1 — output quads   (storage read_write, 36 floats/cubic)
          binding 2 — params uniform (n_cubics u32, padded to 16 bytes)

        Dispatch: ceil(n_cubics / 64) × 1 × 1 workgroups.
        """
        assert self._device is not None
        shader_path = Path(__file__).parent / "shaders" / "cubic_to_quads.wgsl"
        shader_module = self._device.create_shader_module(
            code=shader_path.read_text(encoding="utf-8")
        )

        bgl = self._device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "uniform"},
                },
            ]
        )

        pipeline = self._device.create_compute_pipeline(
            layout=self._device.create_pipeline_layout(bind_group_layouts=[bgl]),
            compute={"module": shader_module, "entry_point": "main"},
        )
        return bgl, pipeline

    def _create_surface_pipeline(
        self,
        proj_bgl: wgpu_t.GPUBindGroupLayout,
        cull_mode: str = "none",
        depth_write: bool = True,
        target_format: str = "bgra8unorm",
        msaa_samples: int = 1,
    ) -> wgpu_t.GPURenderPipeline:
        """Create a surface (mesh) pipeline.

        cull_mode   — WebGPU cull mode passed directly to the pipeline.
                      Use "back" for opaque surfaces (back faces are never
                      visible and culling them halves fragment work).
                      Use "none" for OIT transparent surfaces (both faces
                      must contribute so the interior is visible through
                      the front face).
        depth_write — True for opaque surfaces so they occlude later geometry.
                      False for OIT surfaces so transparent layers do not block
                      each other (they still depth-test against opaque geometry).
        """
        assert self._device is not None
        shader_path = Path(__file__).parent / "shaders" / "surface_combined.wgsl"
        shader_module = self._device.create_shader_module(
            code=shader_path.read_text(encoding="utf-8")
        )
        _blend = {
            "color": {
                "src_factor": "src-alpha",
                "dst_factor": "one-minus-src-alpha",
                "operation": "add",
            },
            "alpha": {
                "src_factor": "one",
                "dst_factor": "one",
                "operation": "add",
            },
        }
        return self._device.create_render_pipeline(
            layout=self._device.create_pipeline_layout(bind_group_layouts=[proj_bgl]),
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [SURFACE_COMBINED_VERTEX_LAYOUT],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": getattr(wgpu.TextureFormat, target_format),
                        "blend": _blend,
                    }
                ],
            },
            primitive={"topology": "triangle-list", "cull_mode": cull_mode},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": depth_write,
                "depth_compare": "less",
                "stencil_front": {
                    "compare": "always",
                    "fail_op": "keep",
                    "depth_fail_op": "keep",
                    "pass_op": "keep",
                },
                "stencil_back": {
                    "compare": "always",
                    "fail_op": "keep",
                    "depth_fail_op": "keep",
                    "pass_op": "keep",
                },
                "stencil_read_mask": 0,
                "stencil_write_mask": 0,
            },
            multisample={
                "count": msaa_samples,
                "mask": 0xFFFF_FFFF,
                "alpha_to_coverage_enabled": False,
            },
        )

    def _create_true_dot_pipeline(
        self,
        proj_bgl: wgpu_t.GPUBindGroupLayout,
        msaa_samples: int = 1,
    ) -> wgpu_t.GPURenderPipeline:
        """Create the TrueDot pipeline (true_dot.wgsl).

        Reuses the camera bind group layout (``proj_bgl``) at group 0.
        Depth write is enabled so dots occlude each other and other geometry.
        Alpha blending is on so the anti-aliased disc edge fades smoothly.
        """
        assert self._device is not None
        shader_path = Path(__file__).parent / "shaders" / "true_dot.wgsl"
        shader = self._device.create_shader_module(
            code=shader_path.read_text(encoding="utf-8")
        )
        _blend = {
            "color": {
                "src_factor": "src-alpha",
                "dst_factor": "one-minus-src-alpha",
                "operation": "add",
            },
            "alpha": {
                "src_factor": "one",
                "dst_factor": "one",
                "operation": "add",
            },
        }
        return self._device.create_render_pipeline(
            layout=self._device.create_pipeline_layout(bind_group_layouts=[proj_bgl]),
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": [TRUE_DOT_VERTEX_LAYOUT],
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{"format": wgpu.TextureFormat.bgra8unorm, "blend": _blend}],
            },
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": "less",
                "stencil_front": {
                    "compare": "always",
                    "fail_op": "keep",
                    "depth_fail_op": "keep",
                    "pass_op": "keep",
                },
                "stencil_back": {
                    "compare": "always",
                    "fail_op": "keep",
                    "depth_fail_op": "keep",
                    "pass_op": "keep",
                },
                "stencil_read_mask": 0,
                "stencil_write_mask": 0,
            },
            multisample={
                "count": msaa_samples,
                "mask": 0xFFFF_FFFF,
                "alpha_to_coverage_enabled": False,
            },
        )

    def _create_image_pipeline(
        self,
        msaa_samples: int = 1,
    ) -> tuple[
        wgpu_t.GPUBindGroupLayout, wgpu_t.GPUBindGroupLayout, wgpu_t.GPURenderPipeline
    ]:
        """Create the render pipeline for ImageMobject textured quads.

        Layout
        ------
        group 0 — camera uniform (reuses ``_proj_bgl``, same as VMobject shaders)
        group 1 — texture_2d<f32> at binding 0, sampler at binding 1
        group 2 — tint uniform: vec3<f32> rgb colour multiplier (16-byte block)

        Vertex buffer (stride 20 B):
          location 0 — in_pos  float32x3  (12 B)
          location 1 — in_uv   float32x2  ( 8 B)
        """
        assert self._device is not None
        assert self._proj_bgl is not None

        shader_path = Path(__file__).parent / "shaders" / "image.wgsl"
        shader = self._device.create_shader_module(
            code=shader_path.read_text(encoding="utf-8")
        )

        # Group 1: texture + sampler
        tex_bgl = self._device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": "float",
                        "view_dimension": "2d",
                        "multisampled": False,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": "filtering"},
                },
            ]
        )

        # Group 2: tint colour uniform (16 bytes: rgb vec3 + 4-byte pad)
        tint_bgl = self._device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform", "min_binding_size": 16},
                },
            ]
        )

        layout = self._device.create_pipeline_layout(
            bind_group_layouts=[self._proj_bgl, tex_bgl, tint_bgl]
        )

        blend = {
            "color": {
                "src_factor": "src-alpha",
                "dst_factor": "one-minus-src-alpha",
                "operation": "add",
            },
            "alpha": {
                "src_factor": "one",
                "dst_factor": "one-minus-src-alpha",
                "operation": "add",
            },
        }

        pipeline = self._device.create_render_pipeline(
            layout=layout,
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": [
                    {
                        "array_stride": 20,  # 3+2 floats × 4 B
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x2", "offset": 12, "shader_location": 1},
                        ],
                    }
                ],
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": wgpu.TextureFormat.bgra8unorm,
                        "blend": blend,
                    }
                ],
            },
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={
                # Must match the render pass's depth format.
                # Images use painter's algorithm (draw order), not depth test.
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": False,
                "depth_compare": "always",
                "stencil_front": {
                    "compare": "always",
                    "fail_op": "keep",
                    "depth_fail_op": "keep",
                    "pass_op": "keep",
                },
                "stencil_back": {
                    "compare": "always",
                    "fail_op": "keep",
                    "depth_fail_op": "keep",
                    "pass_op": "keep",
                },
                "stencil_read_mask": 0,
                "stencil_write_mask": 0,
            },
            multisample={
                "count": msaa_samples,
                "mask": 0xFFFF_FFFF,
                "alpha_to_coverage_enabled": False,
            },
        )

        return tex_bgl, tint_bgl, pipeline

    def _image_fingerprint(self, pixel_array: np.ndarray) -> int:
        """Cheap dirty-check fingerprint for a pixel array.

        Samples the first 64, middle 64, and last 64 bytes of the flattened
        array plus the shape tuple — fast enough for typical image sizes and
        reliably detects in-place changes such as set_opacity().
        """
        flat = pixel_array.ravel()
        n = len(flat)
        if n <= 192:
            return hash((pixel_array.shape, flat.tobytes()))
        mid = n // 2
        sample = np.concatenate([flat[:64], flat[mid : mid + 64], flat[-64:]])
        return hash((pixel_array.shape, sample.tobytes()))

    def _get_image_gpu_resources(
        self, mob: Any
    ) -> tuple[wgpu_t.GPUTexture, wgpu_t.GPUBindGroup] | None:
        """Return (texture, bind_group) for *mob*, re-uploading if pixel_array changed.

        Returns None if the mob has no valid pixel array.
        """
        assert self._device is not None
        assert self._image_tex_bgl is not None

        pixel_array: np.ndarray | None = getattr(mob, "pixel_array", None)
        if pixel_array is None or pixel_array.ndim != 3 or pixel_array.shape[2] < 4:
            return None

        fp = self._image_fingerprint(pixel_array)
        cached = self._image_cache.get(mob)
        if cached is not None and cached[0] == fp:
            return cached[1], cached[2]

        # (Re-)upload texture.
        h, w = pixel_array.shape[:2]
        # Ensure RGBA uint8.
        if pixel_array.dtype != np.uint8:
            pixel_array = pixel_array.astype(np.uint8)

        # bytes_per_row must be a multiple of 256.
        bytes_per_row = w * 4
        aligned_bpr = (bytes_per_row + 255) & ~255
        if aligned_bpr == bytes_per_row:
            data = pixel_array.tobytes()
        else:
            rows = [
                pixel_array[r].ravel().tobytes()
                + b"\x00" * (aligned_bpr - bytes_per_row)
                for r in range(h)
            ]
            data = b"".join(rows)

        tex = self._device.create_texture(
            size=(w, h, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self._device.queue.write_texture(
            {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
            data,
            {"bytes_per_row": aligned_bpr, "rows_per_image": h},
            (w, h, 1),
        )

        sampler = self._device.create_sampler(
            min_filter="linear",
            mag_filter="linear",
            address_mode_u="clamp-to-edge",
            address_mode_v="clamp-to-edge",
        )

        bg = self._device.create_bind_group(
            layout=self._image_tex_bgl,
            entries=[
                {"binding": 0, "resource": tex.create_view()},
                {"binding": 1, "resource": sampler},
            ],
        )

        self._image_cache[mob] = (fp, tex, bg)
        return tex, bg

    def _get_image_vbo(self, mob: Any) -> wgpu_t.GPUBuffer | None:
        """Return a cached 6-vertex (20 B/vertex) VBO for *mob*'s bounding quad.

        The VBO is rebuilt only when mob.points[:4] changes; otherwise the
        same GPUBuffer is reused across frames without any CPU/GPU allocation.

        Corner layout from AbstractImageMobject.reset_points():
          points[0] = UP + LEFT   → UV (0, 0)
          points[1] = UP + RIGHT  → UV (1, 0)
          points[2] = DOWN + LEFT → UV (0, 1)
          points[3] = DOWN + RIGHT→ UV (1, 1)

        Two CCW triangles: [0,1,2] and [1,3,2].

        ``scale_to_resolution`` semantics are enforced at the Mobject level:
        ``AbstractImageMobject.reset_points()`` converts pixel dimensions to
        world-space units using ``scale_to_resolution`` during ``__init__``.
        The renderer reads the resulting world-space corner positions directly
        from ``mob.points`` — no additional scaling is applied here.
        """
        assert self._device is not None

        pts = getattr(mob, "points", None)
        if pts is None or len(pts) < 4:
            return None

        # Fingerprint the 4 corner points exactly (48 bytes — cheap).
        corners = pts[:4].astype(np.float32)
        fp = hash(corners.tobytes())

        cached = self._image_vbo_cache.get(mob)
        if cached is not None and cached[0] == fp:
            return cached[1]

        uvs = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
        )
        idx = [0, 1, 2, 1, 3, 2]
        data = np.empty((6, 5), dtype=np.float32)
        data[:, :3] = corners[idx]
        data[:, 3:] = uvs[idx]

        buf = self._device.create_buffer_with_data(
            data=data.tobytes(),
            usage=wgpu.BufferUsage.VERTEX,
        )
        # Store persistently — NOT in frame_vbos; WeakKeyDictionary releases
        # the buffer when the mob is garbage-collected.
        self._image_vbo_cache[mob] = (fp, buf)
        return buf

    def _get_image_tint_bind_group(self, mob: Any) -> wgpu_t.GPUBindGroup | None:
        """Return a cached GPUBindGroup for *mob*'s tint colour uniform.

        The bind group is rebuilt only when ``mob.color`` changes.  The
        default WHITE tint ``(1, 1, 1)`` is identity — texture is unchanged.
        """
        assert self._device is not None
        assert self._image_tint_bgl is not None

        # Read mob.color → (r, g, b) in [0, 1].  Fall back to white.
        try:
            rgb = np.asarray(mob.color.to_rgb(), dtype=np.float32)
        except Exception:
            rgb = np.ones(3, dtype=np.float32)

        fp = hash(rgb.tobytes())
        cached = self._image_tint_cache.get(mob)
        if cached is not None and cached[0] == fp:
            return cached[2]

        # 16-byte block: rgb (12 B) + 4-byte pad.
        data = np.array([rgb[0], rgb[1], rgb[2], 0.0], dtype=np.float32)
        buf = self._device.create_buffer_with_data(
            data=data.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM,
        )
        bg = self._device.create_bind_group(
            layout=self._image_tint_bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": buf, "offset": 0, "size": 16}}
            ],
        )
        self._image_tint_cache[mob] = (fp, buf, bg)
        return bg

    def _create_oit_resources(self, width: int, height: int) -> None:
        """Create OIT accumulation textures, pipelines, and bind groups."""
        assert self._device is not None
        assert self._proj_bgl is not None

        # ── Accumulation textures ──────────────────────────────────────────
        oit_usage = (
            wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING
        )
        self._oit_accum_texture = self._device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=oit_usage,
        )
        self._oit_accum_view = self._oit_accum_texture.create_view()

        self._oit_reveal_texture = self._device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=oit_usage,
        )
        self._oit_reveal_view = self._oit_reveal_texture.create_view()

        # ── OIT accumulation pipeline ──────────────────────────────────────
        oit_shader_path = Path(__file__).parent / "shaders" / "surface_oit.wgsl"
        oit_shader = self._device.create_shader_module(
            code=oit_shader_path.read_text(encoding="utf-8")
        )
        _accum_blend = {
            "color": {"src_factor": "one", "dst_factor": "one", "operation": "add"},
            "alpha": {"src_factor": "one", "dst_factor": "one", "operation": "add"},
        }
        _reveal_blend = {
            "color": {
                "src_factor": "zero",
                "dst_factor": "one-minus-src-alpha",
                "operation": "add",
            },
            "alpha": {"src_factor": "zero", "dst_factor": "one", "operation": "add"},
        }
        self._surface_oit_pipeline = self._device.create_render_pipeline(
            layout=self._device.create_pipeline_layout(
                bind_group_layouts=[self._proj_bgl]
            ),
            vertex={
                "module": oit_shader,
                "entry_point": "vs_main",
                "buffers": [SURFACE_COMBINED_VERTEX_LAYOUT],
            },
            fragment={
                "module": oit_shader,
                "entry_point": "fs_main",
                "targets": [
                    {"format": wgpu.TextureFormat.rgba16float, "blend": _accum_blend},
                    {"format": wgpu.TextureFormat.rgba16float, "blend": _reveal_blend},
                ],
            },
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": False,
                "depth_compare": "less",
                "stencil_front": {
                    "compare": "always",
                    "fail_op": "keep",
                    "depth_fail_op": "keep",
                    "pass_op": "keep",
                },
                "stencil_back": {
                    "compare": "always",
                    "fail_op": "keep",
                    "depth_fail_op": "keep",
                    "pass_op": "keep",
                },
                "stencil_read_mask": 0,
                "stencil_write_mask": 0,
            },
            multisample={
                "count": 1,
                "mask": 0xFFFF_FFFF,
                "alpha_to_coverage_enabled": False,
            },
        )

        # ── OIT composition pipeline ───────────────────────────────────────
        compose_path = Path(__file__).parent / "shaders" / "oit_compose.wgsl"
        compose_shader = self._device.create_shader_module(
            code=compose_path.read_text(encoding="utf-8")
        )
        self._oit_compose_bgl = self._device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": "unfilterable-float",
                        "view_dimension": "2d",
                        "multisampled": False,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": "unfilterable-float",
                        "view_dimension": "2d",
                        "multisampled": False,
                    },
                },
            ]
        )
        _compose_blend = {
            "color": {
                "src_factor": "src-alpha",
                "dst_factor": "one-minus-src-alpha",
                "operation": "add",
            },
            "alpha": {"src_factor": "one", "dst_factor": "one", "operation": "add"},
        }
        self._oit_compose_pipeline = self._device.create_render_pipeline(
            layout=self._device.create_pipeline_layout(
                bind_group_layouts=[self._oit_compose_bgl]
            ),
            vertex={"module": compose_shader, "entry_point": "vs_main", "buffers": []},
            fragment={
                "module": compose_shader,
                "entry_point": "fs_main",
                "targets": [
                    {"format": wgpu.TextureFormat.bgra8unorm, "blend": _compose_blend}
                ],
            },
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            multisample={
                "count": 1,
                "mask": 0xFFFF_FFFF,
                "alpha_to_coverage_enabled": False,
            },
        )
        self._oit_compose_bind_group = self._device.create_bind_group(
            layout=self._oit_compose_bgl,
            entries=[
                {"binding": 0, "resource": self._oit_accum_view},
                {"binding": 1, "resource": self._oit_reveal_view},
            ],
        )

    # ------------------------------------------------------------------
    # Camera bind group (rebuilt each frame when projection changes)
    # ------------------------------------------------------------------

    def _collect_lights(self) -> list:
        """Return all LightSource instances in the current scene.

        Lights are always added directly to the scene via ``self.add(light)``,
        so scanning the top-level ``scene.mobjects`` list is sufficient and
        avoids a deep O(N_submobjects) traversal through thousands of Surface
        patches every frame.
        """
        if self.scene is None:
            return []
        return [m for m in self.scene.mobjects if isinstance(m, LightSource)]

    _MAX_LIGHTS = 8

    def _pack_camera_uniforms_bytes(
        self,
        proj: np.ndarray,
        view: np.ndarray,
    ) -> bytes:
        """Return a 656-byte camera+lighting uniform payload from explicit proj/view.

        Layout (matches Uniforms struct in surface_combined.wgsl / surface_oit.wgsl):
          offset   0 — projection  mat4x4<f32>  64 B
          offset  64 — view        mat4x4<f32>  64 B
          offset 128 — num_lights  u32           4 B
          offset 132 — _pad        u32 × 3      12 B
          offset 144 — lights      Light × 8   512 B  (each Light = 64 B)
        """
        proj_bytes = proj.T.flatten().astype(np.float32).tobytes()
        view_bytes = view.T.flatten().astype(np.float32).tobytes()

        lights = self._collect_lights()
        n = min(len(lights), self._MAX_LIGHTS)

        # num_lights (u32) + 3× padding u32
        header = np.array([n, 0, 0, 0], dtype=np.uint32).tobytes()

        # Pack up to MAX_LIGHTS light structs; pad the rest with zeros.
        light_data = b""
        for i in range(self._MAX_LIGHTS):
            if i < n:
                light_data += lights[i].pack()
            else:
                light_data += b"\x00" * 64

        return proj_bytes + view_bytes + header + light_data

    # ------------------------------------------------------------------
    # Sub-camera rendering (ZoomedScene / ImageMobjectFromCamera)
    # ------------------------------------------------------------------

    def _sub_camera_proj_view(
        self, sub_cam_frame: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (proj, view) matrices for a MovingCamera's frame viewport.

        The sub-camera is always orthographic.  Its viewport is defined by the
        ``frame`` mobject's current center and size.
        """
        fw = float(sub_cam_frame.get_width())
        fh = float(sub_cam_frame.get_height())
        cen = sub_cam_frame.get_center()
        cx, cy = float(cen[0]), float(cen[1])
        near, far = -100.0, 100.0

        # Orthographic projection using the sub-camera's frame dimensions
        # (centered at origin — the view matrix handles the translation).
        proj = np.array(
            [
                [2.0 / fw, 0.0, 0.0, 0.0],
                [0.0, 2.0 / fh, 0.0, 0.0],
                [0.0, 0.0, -1.0 / (far - near), far / (far - near)],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # View: translate the world so frame center lands at the origin.
        view = np.eye(4, dtype=np.float32)
        view[0, 3] = -cx
        view[1, 3] = -cy
        view[2, 3] = -float(self.camera.focal_distance)
        return proj, view

    def _get_sub_cam_resources(self, mob: Any) -> dict:
        """Return (and lazily create) per-mob GPU resources for sub-camera rendering.

        Returns a dict with keys:
          render_tex, render_view  — rgba8unorm render target
                                     (RENDER_ATTACHMENT | TEXTURE_BINDING | COPY_SRC)
          depth_tex,  depth_view   — depth24plus depth buffer
          uniform_buf              — 656-byte camera uniform buffer (COPY_DST | UNIFORM)
          cam_bg                   — camera-only bind group (proj_bgl, binding 0 = uniform_buf)
          tex_bg                   — image display bind group (image_tex_bgl, binding 0 = render_view)
          staging_buf              — row-aligned COPY_DST | MAP_READ buffer for CPU readback
          staging_aligned_bpr      — aligned bytes-per-row used by the staging buffer
        """
        assert self._device is not None
        assert self._proj_bgl is not None
        assert self._image_tex_bgl is not None

        mob_id = id(mob)
        if mob_id in self._sub_cam_resources:
            return self._sub_cam_resources[mob_id]

        w, h = config.pixel_width, config.pixel_height
        _UBO_SIZE = 656

        # bytes_per_row must be a multiple of 256 for copy_texture_to_buffer.
        aligned_bpr = ((w * 4) + 255) & ~255

        render_tex = self._device.create_texture(
            size=(w, h, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=(
                wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.COPY_SRC
            ),
        )
        render_view = render_tex.create_view()

        depth_tex = self._device.create_texture(
            size=(w, h, 1),
            format=wgpu.TextureFormat.depth24plus,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        depth_view = depth_tex.create_view()

        uniform_buf = self._device.create_buffer(
            size=_UBO_SIZE,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        cam_bg = self._device.create_bind_group(
            layout=self._proj_bgl,
            entries=[
                {
                    "binding": 0,
                    "resource": {"buffer": uniform_buf, "offset": 0, "size": _UBO_SIZE},
                }
            ],
        )

        sampler = self._device.create_sampler(
            min_filter="linear",
            mag_filter="linear",
            address_mode_u="clamp-to-edge",
            address_mode_v="clamp-to-edge",
        )
        tex_bg = self._device.create_bind_group(
            layout=self._image_tex_bgl,
            entries=[
                {"binding": 0, "resource": render_view},
                {"binding": 1, "resource": sampler},
            ],
        )

        # Staging buffer for CPU readback of the rendered sub-camera texture.
        # rgba8unorm — no B↔R swap needed (unlike the main bgra8unorm target).
        staging_buf = self._device.create_buffer(
            size=aligned_bpr * h,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

        resources = {
            "render_tex": render_tex,
            "render_view": render_view,
            "depth_tex": depth_tex,
            "depth_view": depth_view,
            "uniform_buf": uniform_buf,
            "cam_bg": cam_bg,
            "tex_bg": tex_bg,
            "staging_buf": staging_buf,
            "staging_aligned_bpr": aligned_bpr,
        }
        self._sub_cam_resources[mob_id] = resources
        return resources

    def _render_sub_camera_pass(
        self,
        mob: Any,
        encoder: Any,
        normal_fds: list,
    ) -> None:
        """Render a sub-camera view for *mob* (an ImageMobjectFromCamera).

        The sub-camera's view of the main scene is drawn into the mob's
        persistent rgba8unorm render texture using sub-camera pipelines.
        The result is available as ``resources["tex_bg"]`` in the same frame.

        Parameters
        ----------
        mob
            An ``ImageMobjectFromCamera`` instance with a ``camera`` attribute
            that is a ``MovingCamera``.
        encoder
            Active ``GPUCommandEncoder`` (compute pass must already be ended).
        normal_fds
            List of ``_FrameData`` objects from the main frame's normal-camera
            render queue.  The same GPU geometry buffers are reused here with
            a different camera uniform.
        """
        assert self._device is not None
        assert self._fill_stroke_bgl is not None
        assert self._sub_cam_fill_stroke_pipeline is not None
        assert self._sub_cam_fill_stroke_3d_pipeline is not None
        assert self._sub_cam_surface_pipeline is not None

        sub_cam_frame = mob.camera.frame
        proj, view = self._sub_camera_proj_view(sub_cam_frame)
        ubo_bytes = self._pack_camera_uniforms_bytes(proj, view)

        res = self._get_sub_cam_resources(mob)
        self._device.queue.write_buffer(res["uniform_buf"], 0, ubo_bytes)

        bg = self._background_color

        sub_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": res["render_view"],
                    "load_op": "clear",
                    "store_op": "store",
                    "clear_value": tuple(float(c) for c in bg),
                }
            ],
            depth_stencil_attachment={
                "view": res["depth_view"],
                "depth_clear_value": 1.0,
                "depth_load_op": "clear",
                "depth_store_op": "store",
            },
        )

        for fd in normal_fds:
            if fd is None:
                continue
            # Build a sub-camera bind group that combines the sub-camera
            # uniform buffer (binding 0) with the same quads output buffer
            # (binding 1) used in the main render.  This reuses the already-
            # computed quadratic Bezier data without re-running the compute shader.
            if fd.quads_out_buf is not None:
                sub_fill_render_bg = self._device.create_bind_group(
                    layout=self._fill_stroke_bgl,
                    entries=[
                        {
                            "binding": 0,
                            "resource": {
                                "buffer": res["uniform_buf"],
                                "offset": 0,
                                "size": res["uniform_buf"].size,
                            },
                        },
                        {
                            "binding": 1,
                            "resource": {
                                "buffer": fd.quads_out_buf,
                                "offset": 0,
                                "size": fd.quads_out_buf.size,
                            },
                        },
                    ],
                )
            else:
                sub_fill_render_bg = None

            draw_frame_data_subcam(
                sub_pass,
                fd,
                sub_fill_render_bg=sub_fill_render_bg,
                sub_cam_bg=res["cam_bg"],
                fill_2d_pipeline=self._sub_cam_fill_stroke_pipeline,
                fill_3d_pipeline=self._sub_cam_fill_stroke_3d_pipeline,
                surf_pipeline=self._sub_cam_surface_pipeline,
            )

        sub_pass.end()

    # Keep the old name as a shim so any external callers don't break.
    def _pack_camera_uniforms(
        self, proj: np.ndarray, view: np.ndarray
    ) -> wgpu_t.GPUBuffer:
        """Create a throw-away 656-byte uniform buffer (legacy path, rarely used)."""
        assert self._device is not None
        buf = self._device.create_buffer_with_data(
            data=self._pack_camera_uniforms_bytes(proj, view),
            usage=wgpu.BufferUsage.UNIFORM,
        )
        self.frame_vbos.append(buf)
        return buf

    def _build_camera_bind_group(self) -> wgpu_t.GPUBindGroup:
        """Update all three persistent camera uniform buffers for the current frame.

        The three uniform buffers and their bind groups are created once in
        init_scene.  Each frame we write fresh matrix data into the buffers via
        queue.write_buffer so the shaders see the updated camera.

        normal (camera_bind_group)
            Full camera rotation + current projection.  Used for all regular
            mobjects.

        fixed_camera_bind_group
            Rotation-stripped view + current projection.  Used for
            fixed-orientation mobjects: they don't tilt with the camera but
            are still depth-sorted with the rest of the scene.

        fixed_frame_bind_group
            Rotation-stripped view + forced orthographic projection.  Used for
            fixed-in-frame mobjects: 2-D overlays rendered after the 3-D scene
            with a fresh depth buffer so they always appear on top.
        """
        assert self._device is not None
        assert self._camera_uniform_buf is not None
        assert self._fixed_orient_uniform_buf is not None
        assert self._fixed_frame_uniform_buf is not None

        fixed_view = self.camera.fixed_view_matrix

        self._device.queue.write_buffer(
            self._camera_uniform_buf,
            0,
            self._pack_camera_uniforms_bytes(
                self.camera.projection_matrix, self.camera.view_matrix
            ),
        )
        self._device.queue.write_buffer(
            self._fixed_orient_uniform_buf,
            0,
            self._pack_camera_uniforms_bytes(self.camera.projection_matrix, fixed_view),
        )
        self._device.queue.write_buffer(
            self._fixed_frame_uniform_buf,
            0,
            self._pack_camera_uniforms_bytes(
                self.camera.ortho_projection_matrix, fixed_view
            ),
        )

        # Return the persistent normal bind group (unchanged object).
        return self.camera_bind_group

    # ------------------------------------------------------------------
    # Pipeline / device accessors (used by webgpu_vmobject_rendering)
    # ------------------------------------------------------------------

    @property
    def device(self) -> wgpu_t.GPUDevice:
        assert self._device is not None, "init_scene() has not been called"
        return self._device

    @property
    def fill_stroke_pipeline(self) -> wgpu_t.GPURenderPipeline:
        """Combined fill+stroke pipeline — 2-D (no depth write)."""
        assert self._fill_stroke_pipeline is not None, (
            "init_scene() has not been called"
        )
        return self._fill_stroke_pipeline

    @property
    def fill_stroke_3d_pipeline(self) -> wgpu_t.GPURenderPipeline:
        """Combined fill+stroke pipeline — 3-D (depth write + test)."""
        assert self._fill_stroke_3d_pipeline is not None, (
            "init_scene() has not been called"
        )
        return self._fill_stroke_3d_pipeline

    @property
    def fill_stroke_pipeline_1x(self) -> wgpu_t.GPURenderPipeline:
        """Combined fill+stroke pipeline — 2-D, always count=1.

        Used for the fixed-in-frame overlay pass (Pass 4) which renders directly
        into ``_render_texture_view`` after the MSAA resolve has completed.
        """
        assert self._fill_stroke_pipeline_1x is not None, (
            "init_scene() has not been called"
        )
        return self._fill_stroke_pipeline_1x

    @property
    def fill_stroke_3d_pipeline_1x(self) -> wgpu_t.GPURenderPipeline:
        """Combined fill+stroke pipeline — 3-D, always count=1.

        Used for the fixed-in-frame overlay pass (Pass 4) which renders directly
        into ``_render_texture_view`` after the MSAA resolve has completed.
        """
        assert self._fill_stroke_3d_pipeline_1x is not None, (
            "init_scene() has not been called"
        )
        return self._fill_stroke_3d_pipeline_1x

    @property
    def surface_pipeline(self) -> wgpu_t.GPURenderPipeline:
        """Opaque surface pipeline (depth_write=True)."""
        assert self._surface_pipeline is not None, "init_scene() has not been called"
        return self._surface_pipeline

    @property
    def surface_oit_pipeline(self) -> wgpu_t.GPURenderPipeline:
        assert self._surface_oit_pipeline is not None, (
            "init_scene() has not been called"
        )
        return self._surface_oit_pipeline

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------

    def update_frame(
        self,
        scene: Scene,
        mob_list: list | None = None,
        blit_static: bool = False,
        _readback: bool = True,
    ) -> None:
        """Render one frame into the offscreen texture.

        Parameters
        ----------
        mob_list:
            When provided, render only these top-level mobjects instead of
            all of ``scene.mobjects``.  Used by ``save_static_frame_data``
            (static subset) and by ``render`` (moving subset).
        blit_static:
            When True, blit ``_static_texture`` → ``_render_texture`` before
            the main render pass so the static background is preserved.  The
            main pass then uses ``load_op="load"`` to composite moving mobs
            on top.  If False (the default), the frame is cleared to the
            background colour first.

        Pass structure
        --------------
        0. **Texture blit** (when *blit_static*) — copies the pre-rendered
           static layer into the render texture before any render passes.
        1. **Compute pass** — cubic_to_quads.wgsl converts raw cubic Bezier
           control points to quadratic approximations for all three mobject
           groups (normal, fixed-orientation, fixed-in-frame).  This runs
           before any render pass in the same command encoder, so WebGPU's
           implicit pass ordering provides the barrier.
        2. **Main pass** — clears (or loads) the frame; draws normal and
           fixed-orientation mobjects (shared depth buffer; fixed-orient uses
           a rotation-stripped camera bind group).
        3. **OIT accumulation pass** — transparent surfaces use Weighted
           Blended OIT into two rgba16float textures.
        4. **OIT composition pass** — full-screen triangle composites the OIT
           result onto the main texture.
        5. **Fixed-in-frame overlay pass** — 2-D overlays rendered with a
           fresh depth buffer so they always appear on top.
        """
        assert self._device is not None
        assert self._render_texture_view is not None
        assert self._depth_texture_view is not None
        assert self._cubic_to_quads_pipeline is not None

        bg = self._background_color

        # Build all three per-frame camera uniform buffers + bind groups.
        self.camera_bind_group = self._build_camera_bind_group()
        self.frame_vbos = []

        # ── Partition and z-sort mobjects ────────────────────────────────
        cam = self.camera
        fixed_in_frame = cam.fixed_in_frame_mobjects
        fixed_orient = cam.fixed_orientation_mobjects
        fixed_view = self.camera.fixed_view_matrix

        assert self._camera_uniform_buf is not None
        assert self._fixed_orient_uniform_buf is not None
        assert self._fixed_frame_uniform_buf is not None

        if mob_list is not None:
            # Caller (save_static_frame_data, render) already sorted the list.
            source = mob_list
        else:
            # Full-frame path: merge mobjects + foreground_mobjects and apply
            # z_index ordering (Bug 1).  foreground_mobjects are already present
            # in scene.mobjects (add_foreground_mobjects calls add()), so
            # list_update just removes the duplicates from the left side.
            # We then sort with a two-key tuple so that:
            #   key[0] = 0 for normal mobs, 1 for foreground mobs
            #   key[1] = z_index
            # This ensures foreground mobs always draw last (on top) even when
            # they share z_index=0 with regular mobs (Bug 3).
            all_mobs = list_update(
                list(scene.mobjects), list(scene.foreground_mobjects)
            )
            if self.camera.use_z_index:
                foreground_ids = {id(m) for m in scene.foreground_mobjects}
                source = sorted(
                    all_mobs,
                    key=lambda m: (1 if id(m) in foreground_ids else 0, m.z_index),
                )
            else:
                source = all_mobs

        # Build a z-ordered render queue by walking `source` in order.
        #
        # Rules:
        #   • fixed_in_frame mobs → skipped here, collected separately below
        #   • fixed_orient mobs   → VMobject batch with stripped-rotation camera
        #   • normal VMobjects    → VMobject batch with full camera
        #   • ImageMobjects       → image draw item, flushing any pending
        #                           VMobject runs first so z-order is respected
        #   • containers (Group…) → recursed
        #
        # The resulting queue is a list of items:
        #   ('vmobs', _FrameData, camera_bind_group)
        #   ('image', ImageMobject)
        #
        # Within the main render pass these are drawn in queue order, giving
        # correct painter's-algorithm depth for any interleaving of images and
        # VMobjects in scene.mobjects.

        render_queue: list[tuple] = []
        _run_normal: list = []
        _run_orient: list = []
        _seen: set[int] = set()

        def _flush_runs() -> None:
            if _run_normal:
                fd = collect_frame_data(
                    self,
                    list(_run_normal),
                    self._camera_uniform_buf,
                    cache_slot="normal",
                )
                if fd is not None:
                    render_queue.append(("vmobs", fd, self.camera_bind_group))
                _run_normal.clear()
            if _run_orient:
                fd = collect_frame_data(
                    self,
                    list(_run_orient),
                    self._fixed_orient_uniform_buf,
                    view_matrix_override=fixed_view,
                    center_view_matrix=self.camera.view_matrix,
                    cache_slot="orient",
                )
                if fd is not None:
                    render_queue.append(("vmobs", fd, self.fixed_camera_bind_group))
                _run_orient.clear()

        def _walk(mob: Any) -> None:
            if id(mob) in _seen:
                return
            _seen.add(id(mob))
            if isinstance(mob, AbstractImageMobject):
                _flush_runs()
                render_queue.append(("image", mob))
                # Recurse into submobjects (e.g. the display_frame SurroundingRectangle
                # added by ImageMobjectFromCamera.add_display_frame()) so they are
                # rendered as VMobject overlays on top of the image quad.
                for sub in mob.submobjects:
                    _walk(sub)
            elif isinstance(mob, DotCloud3D):
                # WebGPU dot cloud — rendered as screen-aligned sphere quads.
                _flush_runs()
                render_queue.append(("truedot", mob))
            elif isinstance(mob, Surface):
                # Parametric Surface: add individually so collect_frame_data
                # uses the Surface lighting/geometry cache path.
                if mob in fixed_in_frame:
                    pass  # handled in overlay pass below
                elif mob in fixed_orient:
                    _run_orient.append(mob)
                else:
                    _run_normal.append(mob)
            elif isinstance(mob, VMobject):
                # Container check: if any *direct* child is a Surface, Image, or
                # DotCloud, recurse rather than treating this mob as a monolithic
                # VMobject.  This ensures e.g. VGroup(Sphere(), Sphere()) routes
                # each Sphere through the surface rendering path.
                if mob.submobjects and any(
                    isinstance(s, (Surface, AbstractImageMobject, DotCloud3D))
                    for s in mob.submobjects
                ):
                    for sub in mob.submobjects:
                        _walk(sub)
                elif mob in fixed_in_frame:
                    pass  # handled in overlay pass below
                elif mob in fixed_orient:
                    _run_orient.append(mob)
                else:
                    _run_normal.append(mob)
            else:
                for sub in mob.submobjects:
                    _walk(sub)

        for mob in source:
            _walk(mob)
        _flush_runs()

        # Pre-fetch image GPU resources (texture upload, VBO) before the
        # command encoder starts.  Replace ('image', mob) queue items with
        # ('image', vbo, tex_bg) so the render loop has no CPU work left.
        # Similarly expand TrueDot mobs into vertex arrays and GPU buffers.
        #
        # ImageMobjectFromCamera mobs are handled specially: instead of reading
        # their pixel_array (which is never populated by the WebGPU renderer),
        # we use the pre-rendered sub-camera texture produced in Pass 0.5.
        resolved_queue: list[tuple] = []
        for item in render_queue:
            if item[0] == "image":
                mob = item[1]
                vbo = self._get_image_vbo(mob)
                tint_bg = self._get_image_tint_bind_group(mob)
                if isinstance(mob, ImageMobjectFromCamera):
                    res = self._sub_cam_resources.get(id(mob))
                    tex_bg = res["tex_bg"] if res is not None else None
                    if vbo is not None and tex_bg is not None and tint_bg is not None:
                        resolved_queue.append(("image", vbo, tex_bg, tint_bg))
                else:
                    resources = self._get_image_gpu_resources(mob)
                    if (
                        vbo is not None
                        and resources is not None
                        and tint_bg is not None
                    ):
                        resolved_queue.append(("image", vbo, resources[1], tint_bg))
            elif item[0] == "truedot":
                mob = item[1]
                arr = build_true_dot_vbo(mob)
                if arr is not None and len(arr) > 0:
                    buf = self._device.create_buffer_with_data(
                        data=arr.tobytes(),
                        usage=wgpu.BufferUsage.VERTEX,
                    )
                    self.frame_vbos.append(buf)
                    resolved_queue.append(("truedot", buf, len(arr)))
            else:
                resolved_queue.append(item)

        # Fixed-in-frame: always last, separate overlay pass.
        fixed_frame_mobs = [
            m
            for m in _seen
            if False  # placeholder — rebuilt below from source flatten
        ]

        # Re-flatten source to get all VMobjects (including those inside containers)
        # and filter to the fixed_in_frame set.
        def _flatten_vmobjects(src: list) -> list:
            out: list = []
            seen2: set[int] = set()

            def _f(m: Any) -> None:
                if id(m) in seen2:
                    return
                seen2.add(id(m))
                if isinstance(m, VMobject):
                    out.append(m)
                else:
                    for s in m.submobjects:
                        _f(s)

            for m in src:
                _f(m)
            return out

        fixed_frame_mobs = [
            m for m in _flatten_vmobjects(source) if m in fixed_in_frame
        ]
        fixed_frame_fd = collect_frame_data(
            self,
            fixed_frame_mobs,
            self._fixed_frame_uniform_buf,
            view_matrix_override=fixed_view,
            proj_matrix_override=self.camera.ortho_projection_matrix,
            cache_slot="frame",
        )

        # OIT surfaces come from all normal VMobject batches in the queue.
        all_normal_fds = [
            item[1]
            for item in resolved_queue
            if item[0] == "vmobs" and item[2] is self.camera_bind_group
        ]

        encoder = self._device.create_command_encoder()

        # ── Pre-pass: blit static background ─────────────────────────────
        # When compositing moving mobs on top of the pre-rendered static layer,
        # copy the static texture into the render texture before any render
        # passes.  The subsequent main pass uses load_op="load" so the static
        # pixels are preserved under the newly drawn moving mobs.
        #
        # With MSAA the static optimisation is bypassed: the MSAA texture is an
        # intermediate buffer that always resolves into _render_texture at the
        # end of Pass 1, overwriting whatever was there.  Every frame is fully
        # re-rendered instead.  (MSAA already implies a quality-over-speed
        # trade-off, so the extra work per frame is acceptable.)
        _do_static_blit = (
            blit_static
            and self._has_static_frame
            and self._static_texture is not None
            and self._msaa_samples == 1
        )
        if _do_static_blit:
            encoder.copy_texture_to_texture(
                {"texture": self._static_texture, "mip_level": 0, "origin": (0, 0, 0)},
                {"texture": self._render_texture, "mip_level": 0, "origin": (0, 0, 0)},
                (config.pixel_width, config.pixel_height, 1),
            )

        # ── Pass 0: compute — cubic → quadratic conversion ────────────────
        # Runs before any render pass; WebGPU guarantees the output buffer is
        # ready by the time the fragment shader reads it in Pass 1.
        cp = encoder.begin_compute_pass()
        cp.set_pipeline(self._cubic_to_quads_pipeline)
        all_fds = [item[1] for item in resolved_queue if item[0] == "vmobs"] + (
            [fixed_frame_fd] if fixed_frame_fd is not None else []
        )
        for fd in all_fds:
            if fd.n_cubics_total > 0 and fd.compute_bg is not None:
                cp.set_bind_group(0, fd.compute_bg, [], 0, 0)
                cp.dispatch_workgroups((fd.n_cubics_total + 63) // 64, 1, 1)
        cp.end()

        # ── Pass 0.5: sub-camera render passes ───────────────────────────
        # Render scene geometry into each ImageMobjectFromCamera's private
        # rgba8unorm texture so it is ready to be sampled as an image in
        # Pass 1.  These passes share the already-computed quads buffers
        # from the compute pass above; no re-dispatch is needed.
        if self.camera.image_mobjects_from_cameras:
            normal_fds = [item[1] for item in resolved_queue if item[0] == "vmobs"]
            for sub_mob in self.camera.image_mobjects_from_cameras:
                self._get_sub_cam_resources(sub_mob)  # ensure resources created
                self._render_sub_camera_pass(sub_mob, encoder, normal_fds)

            # Encode texture → staging-buffer copies so that after submit the
            # rendered sub-camera pixels are available for CPU readback.
            # This allows ImageMobjectFromCamera.get_pixel_array() to return
            # current frame data instead of the stale initial pixel_array.
            w, h = config.pixel_width, config.pixel_height
            for sub_mob in self.camera.image_mobjects_from_cameras:
                res = self._sub_cam_resources.get(id(sub_mob))
                if res is None:
                    continue
                aligned_bpr = res["staging_aligned_bpr"]
                encoder.copy_texture_to_buffer(
                    {"texture": res["render_tex"], "mip_level": 0, "origin": (0, 0, 0)},
                    {
                        "buffer": res["staging_buf"],
                        "offset": 0,
                        "bytes_per_row": aligned_bpr,
                        "rows_per_image": h,
                    },
                    (w, h, 1),
                )

        # ── Pass 1: main render ───────────────────────────────────────────
        # Draw the z-ordered render queue (VMobject batches and images
        # interleaved in scene.mobjects order) so painter's-algorithm depth
        # is respected for any combination of images and geometry.
        #
        # MSAA path (msaa_samples > 1):
        #   color view    — _msaa_texture_view (intermediate MSAA buffer)
        #   resolve_target — _render_texture_view (receives the resolved pixels)
        #   store_op      — "discard" for the MSAA buffer (transient, never read back)
        #   depth view    — _msaa_depth_texture_view (sample_count must match pipeline)
        #   depth store   — "discard" (MSAA depth is not sampled later; OIT uses
        #                   _depth_texture_view at count=1 separately)
        #
        # Non-MSAA path (msaa_samples == 1):
        #   color view    — _render_texture_view directly
        #   depth view    — _depth_texture_view (reused by OIT pass below)
        if self._msaa_samples > 1:
            assert self._msaa_texture_view is not None
            assert self._msaa_depth_texture_view is not None
            _color_attachment = {
                "view": self._msaa_texture_view,
                "resolve_target": self._render_texture_view,
                "load_op": "clear",
                "store_op": "discard",
                "clear_value": tuple(float(c) for c in bg),
            }
            _depth_attachment = {
                "view": self._msaa_depth_texture_view,
                "depth_clear_value": 1.0,
                "depth_load_op": "clear",
                "depth_store_op": "discard",
            }
        else:
            color_load_op = "load" if _do_static_blit else "clear"
            _color_attachment = {
                "view": self._render_texture_view,
                "load_op": color_load_op,
                "store_op": "store",
                "clear_value": tuple(float(c) for c in bg),
            }
            _depth_attachment = {
                "view": self._depth_texture_view,
                "depth_clear_value": 1.0,
                "depth_load_op": "clear",
                "depth_store_op": "store",
            }

        main_pass = encoder.begin_render_pass(
            color_attachments=[_color_attachment],
            depth_stencil_attachment=_depth_attachment,
        )
        self.current_render_pass = main_pass

        current_pipeline = None
        for item in resolved_queue:
            if item[0] == "image":
                _, vbo, tex_bg, tint_bg = item
                if current_pipeline != "image":
                    main_pass.set_pipeline(self._image_pipeline)
                    main_pass.set_bind_group(0, self.camera_bind_group, [], 0, 0)
                    current_pipeline = "image"
                main_pass.set_bind_group(1, tex_bg, [], 0, 0)
                main_pass.set_bind_group(2, tint_bg, [], 0, 0)
                main_pass.set_vertex_buffer(0, vbo)
                main_pass.draw(6)
            elif item[0] == "truedot":
                _, buf, n_verts = item
                if current_pipeline != "truedot":
                    main_pass.set_pipeline(self._true_dot_pipeline)
                    main_pass.set_bind_group(0, self.camera_bind_group, [], 0, 0)
                    current_pipeline = "truedot"
                main_pass.set_vertex_buffer(0, buf)
                main_pass.draw(n_verts)
            elif item[0] == "vmobs":
                _, fd, cam_bg = item
                current_pipeline = "vmobs"  # draw_frame_data sets its own pipelines
                draw_frame_data(self, fd, cam_bg)

        main_pass.end()

        # ── Pass 2: OIT accumulation ──────────────────────────────────────
        # Collect OIT surfaces from all normal-camera VMobject batches.
        oit_fds = [fd for fd in all_normal_fds if fd.oit_indices]
        if oit_fds:
            # Use the first fd with OIT surfaces as representative; actual
            # OIT draw loops over all of them below.
            oit_fd = oit_fds[0]
        else:
            oit_fd = None
        if oit_fds:
            # When MSAA is enabled the opaque depth lives in _msaa_depth_texture
            # (sample_count=N).  _depth_texture (count=1) was not written in
            # Pass 1, so OIT transparent surfaces cannot depth-test against
            # opaque geometry — use "clear" to avoid reading stale data.
            # In practice this means transparent surfaces are not clipped by
            # opaque geometry when MSAA is on, which is acceptable for the
            # typical use case (transparent surfaces in 3-D scenes).
            oit_depth_load_op = "clear" if self._msaa_samples > 1 else "load"
            oit_pass = encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": self._oit_accum_view,
                        "load_op": "clear",
                        "store_op": "store",
                        "clear_value": (0.0, 0.0, 0.0, 0.0),
                    },
                    {
                        "view": self._oit_reveal_view,
                        "load_op": "clear",
                        "store_op": "store",
                        "clear_value": (1.0, 1.0, 1.0, 1.0),
                    },
                ],
                depth_stencil_attachment={
                    "view": self._depth_texture_view,
                    "depth_load_op": oit_depth_load_op,
                    "depth_store_op": "discard",
                },
            )
            oit_pass.set_pipeline(self.surface_oit_pipeline)
            oit_pass.set_bind_group(0, self.camera_bind_group, [], 0, 0)
            for oit_fd in oit_fds:
                for idx in oit_fd.oit_indices:
                    arr = oit_fd.surface_parts[idx]
                    oit_pass.set_vertex_buffer(
                        0,
                        oit_fd.surface_buf,
                        oit_fd.surface_byte_offsets[idx],
                        arr.nbytes,
                    )
                    oit_pass.draw(len(arr), 1, 0, 0)
            oit_pass.end()

            # ── Pass 3: OIT composition ──────────────────────────────────
            compose_pass = encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": self._render_texture_view,
                        "load_op": "load",
                        "store_op": "store",
                    }
                ],
            )
            compose_pass.set_pipeline(self._oit_compose_pipeline)
            compose_pass.set_bind_group(0, self._oit_compose_bind_group, [], 0, 0)
            compose_pass.draw(3, 1, 0, 0)
            compose_pass.end()

        # ── Pass 4: fixed-in-frame overlay ───────────────────────────────
        # Rendered after OIT so overlays always appear on top of the 3-D scene.
        # Fresh depth buffer: overlays only depth-test against each other.
        #
        # This pass always renders directly into _render_texture_view at
        # sample_count=1, regardless of the MSAA setting.  The MSAA resolve
        # (end of Pass 1) has already completed, so _render_texture contains
        # the full scene.  Fixed-in-frame mobjects are 2-D overlays whose SDF
        # anti-aliasing is already excellent; MSAA adds nothing here.
        #
        # When MSAA is enabled the fill+stroke pipelines have multisample
        # count=N and cannot be used in a count=1 pass.  We temporarily swap
        # in the _1x pipeline variants so that draw_frame_data issues
        # compatible GPU commands.
        if fixed_frame_fd is not None:
            fixed_pass = encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": self._render_texture_view,
                        "load_op": "load",
                        "store_op": "store",
                    }
                ],
                depth_stencil_attachment={
                    "view": self._depth_texture_view,
                    "depth_clear_value": 1.0,
                    "depth_load_op": "clear",
                    "depth_store_op": "discard",
                },
            )
            self.current_render_pass = fixed_pass
            if self._msaa_samples > 1:
                # Temporarily expose count=1 pipelines so draw_frame_data
                # records compatible draw calls for this count=1 pass.
                _saved_2d = self._fill_stroke_pipeline
                _saved_3d = self._fill_stroke_3d_pipeline
                self._fill_stroke_pipeline = self._fill_stroke_pipeline_1x
                self._fill_stroke_3d_pipeline = self._fill_stroke_3d_pipeline_1x
            draw_frame_data(self, fixed_frame_fd, self.fixed_frame_bind_group)
            if self._msaa_samples > 1:
                self._fill_stroke_pipeline = _saved_2d
                self._fill_stroke_3d_pipeline = _saved_3d
            fixed_pass.end()

        # ── Invalidate readback cache ────────────────────────────────────
        # A new frame is being rendered; cached pixel data from the previous
        # frame is no longer valid.
        self._readback_cache = None

        # ── Pool-based pre-staged readback ───────────────────────────────
        # Append the readback compute dispatch + buffer copy into the current
        # pool slot, then call map_async immediately after submit.  The GPU
        # processes the readback asynchronously while the CPU continues.
        # When get_image() / get_frame() dequeues the slot, it calls
        # sync_wait() — which returns instantly when the pool has been filled
        # (the oldest slot has been in-flight for >= _READBACK_POOL frames).
        #
        # Guard: only pre-stage if the pool has a free slot.  A slot is free
        # once it has been read (unmap called) by _get_mapped_frame_array.
        # If all slots are still in-flight we skip pre-staging for this frame;
        # _get_mapped_frame_array falls back to a fresh submit+map_sync.
        #
        # _readback=False skips readback staging entirely (used by
        # save_static_frame_data, which renders for texture capture only and
        # must not pollute the pool with non-movie frames).
        if (
            _readback
            and self._readback_compute_pipeline is not None
            and self._readback_compute_bind_group is not None
            and self._readback_storage_buf is not None
            and self._readback_pool
            and len(self._readback_queue) < self._READBACK_POOL
        ):
            width = config.pixel_width
            height = config.pixel_height
            packed_size = width * height * 4
            slot = self._readback_write_slot

            cp = encoder.begin_compute_pass()
            cp.set_pipeline(self._readback_compute_pipeline)
            cp.set_bind_group(0, self._readback_compute_bind_group)
            cp.dispatch_workgroups((width + 15) // 16, (height + 15) // 16)
            cp.end()

            encoder.copy_buffer_to_buffer(
                self._readback_storage_buf,
                0,
                self._readback_pool[slot],
                0,
                packed_size,
            )

            self._device.queue.submit([encoder.finish()])

            self._readback_queue.append(slot)
            self._readback_write_slot = (slot + 1) % self._READBACK_POOL
        else:
            self._device.queue.submit([encoder.finish()])

        # ── Sub-camera CPU readback ───────────────────────────────────────
        # Populate mob.camera.pixel_array from the staged sub-camera texture
        # data so that ImageMobjectFromCamera.get_pixel_array() returns the
        # current frame rather than the stale initial array.
        # rgba8unorm needs no B↔R channel swap (unlike the main bgra8unorm target).
        if self.camera.image_mobjects_from_cameras:
            w, h = config.pixel_width, config.pixel_height
            for sub_mob in self.camera.image_mobjects_from_cameras:
                res = self._sub_cam_resources.get(id(sub_mob))
                if res is None:
                    continue
                staging_buf = res["staging_buf"]
                aligned_bpr = res["staging_aligned_bpr"]
                staging_buf.map_sync(wgpu.MapMode.READ)
                raw = bytes(staging_buf.read_mapped())
                staging_buf.unmap()
                if aligned_bpr == w * 4:
                    arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4).copy()
                else:
                    # Strip row padding before reshaping.
                    rows = [
                        raw[r * aligned_bpr : r * aligned_bpr + w * 4] for r in range(h)
                    ]
                    arr = (
                        np.frombuffer(b"".join(rows), dtype=np.uint8)
                        .reshape(h, w, 4)
                        .copy()
                    )
                # Write into the Cairo sub-camera's pixel_array so that
                # ImageMobjectFromCamera.get_pixel_array() returns current data.
                try:
                    sub_mob.camera.pixel_array = arr
                except Exception:
                    pass

        self.current_render_pass = None
        # camera_bind_group is now persistent (created once in init_scene) —
        # do NOT null it here.
        self.frame_vbos = []

        self.animation_elapsed_time = time.time() - self.animation_start_time

    # ------------------------------------------------------------------
    # Frame readback
    # ------------------------------------------------------------------

    def _create_readback_pipeline(self, width: int, height: int) -> None:
        """Create the compact-readback compute pipeline and its persistent buffers.

        The compute shader (readback_compact.wgsl) reads every pixel from the
        bgra8unorm render texture and writes tightly-packed RGBA u32 values into
        a storage buffer — eliminating two CPU operations that previously ran on
        every frame:

          * Row-padding strip  — copy_texture_to_buffer requires bytes_per_row
            to be a multiple of 256; the shader writes directly to tight index
            ``y * width + x``, so the output is already compact.

          * B↔R channel swap  — textureLoad() returns components as (r, g, b, a)
            regardless of the bgra physical layout, so the output is already in
            RGBA byte order.
        """
        assert self._device is not None
        assert self._render_texture_view is not None

        shader_path = Path(__file__).parent / "shaders" / "readback_compact.wgsl"
        shader = self._device.create_shader_module(
            code=shader_path.read_text(encoding="utf-8")
        )

        self._readback_compute_bgl = self._device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "texture": {
                        "sample_type": "float",
                        "view_dimension": "2d",
                        "multisampled": False,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
            ]
        )

        self._readback_compute_pipeline = self._device.create_compute_pipeline(
            layout=self._device.create_pipeline_layout(
                bind_group_layouts=[self._readback_compute_bgl]
            ),
            compute={"module": shader, "entry_point": "main"},
        )

        packed_size = width * height * 4
        self._readback_storage_buf = self._device.create_buffer(
            size=packed_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        # Allocate all pool slots up front — no per-frame allocation ever.
        self._readback_pool = [
            self._device.create_buffer(
                size=packed_size,
                usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
            )
            for _ in range(self._READBACK_POOL)
        ]
        self._readback_compute_bind_group = self._device.create_bind_group(
            layout=self._readback_compute_bgl,
            entries=[
                {"binding": 0, "resource": self._render_texture_view},
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self._readback_storage_buf,
                        "offset": 0,
                        "size": packed_size,
                    },
                },
            ],
        )

    def _get_mapped_frame_array(self) -> np.ndarray:
        """Return the current frame as a flat uint8 numpy array (H*W*4 elements).

        Cache path (fastest): update_frame() has not been called since the last
        readback, so the frame is unchanged — return the cached array directly
        with zero GPU interaction.

        Pool path: update_frame() enqueued a slot index into _readback_queue.
        We dequeue the oldest slot and call map_sync().  When the pool has
        been filled (_READBACK_POOL frames submitted before the first read),
        the GPU work is already complete; map_sync() only pays the driver's
        buffer-mapping overhead (< 1 ms) rather than a full GPU pipeline
        stall.

        Fallback path: nothing is in the queue (get_image called without a
        preceding update_frame, or the pool was full at submit time).  We
        submit fresh readback work to the next pool slot and block with
        map_sync — same behaviour as before, but still using pool buffers so
        no new GPU allocation occurs.
        """
        assert self._device is not None
        assert self._readback_compute_pipeline is not None
        assert self._readback_compute_bind_group is not None
        assert self._readback_storage_buf is not None
        assert self._readback_pool

        # ── Cache hit ────────────────────────────────────────────────────
        if self._readback_cache is not None:
            return self._readback_cache

        width = config.pixel_width
        height = config.pixel_height
        packed_size = width * height * 4

        if self._readback_queue:
            # ── Pool path: dequeue oldest in-flight slot ─────────────────
            # The slot was submitted >= _READBACK_POOL frames ago, so the GPU
            # has had enough time to finish.  map_sync initiates the mapping
            # and waits; since the GPU work is already done, only the driver's
            # buffer-mapping overhead remains (typically < 1 ms).
            slot = self._readback_queue.popleft()
            buf = self._readback_pool[slot]
            buf.map_sync(wgpu.MapMode.READ)
        else:
            # ── Fallback path: submit now, block ─────────────────────────
            slot = self._readback_write_slot
            buf = self._readback_pool[slot]

            encoder = self._device.create_command_encoder()

            cp = encoder.begin_compute_pass()
            cp.set_pipeline(self._readback_compute_pipeline)
            cp.set_bind_group(0, self._readback_compute_bind_group)
            cp.dispatch_workgroups((width + 15) // 16, (height + 15) // 16)
            cp.end()

            encoder.copy_buffer_to_buffer(
                self._readback_storage_buf,
                0,
                buf,
                0,
                packed_size,
            )

            self._device.queue.submit([encoder.finish()])
            buf.map_sync(wgpu.MapMode.READ)
            self._readback_write_slot = (slot + 1) % self._READBACK_POOL

        # numpy copy is ~3× faster than bytes() for large buffers (SIMD path).
        arr = np.frombuffer(buf.read_mapped(), dtype=np.uint8).copy()
        buf.unmap()

        self._readback_cache = arr
        return arr

    def get_image(self) -> Image.Image:
        """Return the current frame as a PIL Image (RGBA)."""
        arr = self._get_mapped_frame_array()
        return Image.fromarray(
            arr.reshape(config.pixel_height, config.pixel_width, 4), "RGBA"
        )

    def get_frame(self) -> np.ndarray:
        """Return the current frame as a (height, width, 4) uint8 NumPy array."""
        return self._get_mapped_frame_array().reshape(
            config.pixel_height, config.pixel_width, 4
        )

    # ------------------------------------------------------------------
    # Window helpers
    # ------------------------------------------------------------------

    def should_create_window(self) -> bool:
        """Mirror of ``OpenGLRenderer.should_create_window``.

        A preview window is opened when ``--preview`` is active and the
        renderer is not writing a movie or saving a still frame.
        """
        if config["force_window"]:
            logger.warning(
                "'--force_window' is enabled; this is intended for debugging "
                "and may impact performance when combined with file output.",
            )
            return True
        return (
            config["preview"]
            and not config["save_last_frame"]
            and not config["format"]
            and not config["write_to_movie"]
            and not config["dry_run"]
        )

    def pixel_coords_to_space_coords(
        self,
        px: float,
        py: float,
        relative: bool = False,
        top_left: bool = False,
    ) -> np.ndarray:
        """Convert pixel coordinates to Manim scene-space coordinates.

        Parameters
        ----------
        px, py:
            Pixel position.  For ``relative=False``, these are absolute
            pixel coordinates within the render texture.
        relative:
            When True, treat *px*/*py* as a delta and return the
            corresponding scene-space delta (normalised to ``[-1, 1]``
            then scaled).
        top_left:
            When True (the default for ``rendercanvas``), the origin is
            at the top-left corner; y increases downward.
        """
        pixel_width = config.pixel_width
        pixel_height = config.pixel_height
        frame_height = config.frame_height
        frame_center = self.camera.get_center()

        if relative:
            return 2.0 * np.array([px / pixel_width, py / pixel_height, 0.0])

        scale = frame_height / pixel_height
        y_direction = -1 if top_left else 1
        return frame_center + scale * np.array(
            [(px - pixel_width / 2), y_direction * (py - pixel_height / 2), 0.0]
        )

    # ------------------------------------------------------------------
    # Scene rendering
    # ------------------------------------------------------------------

    def render(self, scene: Scene, frame_offset: float, moving_mobjects: list) -> None:
        if self._has_static_frame:
            # Use the family-level moving list produced by begin_animations()
            # directly.  That list is already z_index-sorted by
            # extract_mobject_family_members and is at the correct granularity
            # (same as what Cairo passes to its camera).  Filtering
            # scene.mobjects by static IDs was wrong because it operated at
            # top-level container granularity while _static_mob_ids stores
            # family-member IDs (Bug 2 fix).
            self.update_frame(scene, mob_list=list(moving_mobjects), blit_static=True)
        else:
            self.update_frame(scene)
        if self.skip_animations:
            return
        self.file_writer.write_frame(self)
        if self.window is not None:
            if self.window.is_closing:
                self.window = None
                return
            self.window.present()
            while self.animation_elapsed_time < frame_offset:
                if self.window.is_closing:
                    self.window = None
                    break
                if self._has_static_frame:
                    self.update_frame(
                        scene, mob_list=list(moving_mobjects), blit_static=True
                    )
                else:
                    self.update_frame(scene)
                self.window.present()

    def play(self, scene: Scene, *animations: Any, **kwargs: Any) -> None:
        self.animation_start_time = time.time()
        self.skip_animations = self._original_skipping_status
        self.update_skipping_status()

        # Compile first so we can compute a real hash (same order as CairoRenderer).
        scene.compile_animation_data(*animations, **kwargs)

        if self.skip_animations:
            hash_current_animation = None
            self.time += scene.duration
        elif config["disable_caching"]:
            hash_current_animation = f"uncached_{self.num_plays:05}"
        else:
            assert scene.animations is not None
            hash_current_animation = get_hash_from_play_call(
                scene, self.camera, scene.animations, scene.mobjects
            )
            if self.file_writer.is_already_cached(hash_current_animation):
                logger.info(
                    "Animation %d: using cached data (hash: %s)",
                    self.num_plays,
                    hash_current_animation,
                )
                self.skip_animations = True
                self.time += scene.duration

        self.animations_hashes.append(hash_current_animation)
        self.file_writer.add_partial_movie_file(hash_current_animation)

        self.file_writer.begin_animation(not self.skip_animations)
        scene.begin_animations()

        # Pre-render static mobjects once, matching Cairo's optimisation.
        # scene.static_mobjects is populated by begin_animations() above.
        self.save_static_frame_data(scene, scene.static_mobjects)

        if scene.is_current_animation_frozen_frame():
            self.update_frame(scene)
            if not self.skip_animations:
                self.file_writer.write_frame(
                    self, num_frames=int(config.frame_rate * scene.duration)
                )
            if self.window is not None:
                if self.window.is_closing:
                    self.window = None
                else:
                    self.window.present()
                    while time.time() - self.animation_start_time < scene.duration:
                        if self.window.is_closing:
                            self.window = None
                            break
                        self.window.present()
            self.animation_elapsed_time = scene.duration
        else:
            scene.play_internal()

        self.file_writer.end_animation(not self.skip_animations)
        self.time += scene.duration
        self.num_plays += 1

    def scene_finished(self, scene: Scene) -> None:
        if self.num_plays > 0:
            self.file_writer.finish()
        elif self.num_plays == 0 and config.write_to_movie:
            config.save_last_frame = True
            config.write_to_movie = False

        if self._should_save_last_frame():
            config.save_last_frame = True
            self.update_frame(scene)
            self.file_writer.save_image(self.get_image())

        # Explicitly close the preview window so that GlfwRenderCanvas._rc_close()
        # destroys the GLFW window handle while GLFW is still alive.  Without this,
        # Python shutdown calls GlfwCanvasGroup.__del__ → glfw.terminate() first,
        # then the GlfwRenderCanvas.__del__ fires and tries glfw.destroy_window()
        # on an already-terminated GLFW library, producing a GLFWError warning.
        if self.window is not None:
            try:
                self.window.destroy()
            except Exception:
                pass
            self.window = None

    def save_static_frame_data(self, scene: Scene, static_mobjects: Any) -> None:
        """Render *static_mobjects* once and cache the result in ``_static_texture``.

        Called by ``play()`` after ``begin_animations()``, before the per-frame
        loop starts.  Subsequent calls to ``render()`` blit this cached texture
        as the background and only re-draw the moving subset, matching Cairo's
        static-image compositing optimisation.

        When *static_mobjects* is empty (all mobs are moving, or no mobs at all),
        the static frame is cleared so that ``render()`` falls back to a full
        redraw each frame.
        """
        assert self._device is not None
        assert self._static_texture is not None

        static_list = list(static_mobjects) if static_mobjects else []

        if not static_list:
            self._has_static_frame = False
            self._static_mob_ids = set()
            return

        # If the camera itself is animated (e.g. begin_ambient_camera_rotation),
        # any pre-rendered static texture is stale on the very next frame because
        # the projection changes.  Disable the optimisation so every frame is
        # fully re-rendered with the correct camera orientation.
        if self.camera.has_time_based_updater():
            self._has_static_frame = False
            self._static_mob_ids = set()
            return

        self._static_mob_ids = set(id(m) for m in static_list)

        # Render the static mob list into _render_texture (full clear + draw).
        # _readback=False: this render is for texture capture only — it must not
        # enqueue a readback pool slot, because save_static_frame_data is not
        # writing a movie frame and a stale slot in the pool would cause the next
        # write_frame() call to read wrong pixel data.
        self.update_frame(
            scene, mob_list=static_list, blit_static=False, _readback=False
        )

        # Copy _render_texture → _static_texture for later per-frame blits.
        encoder = self._device.create_command_encoder()
        encoder.copy_texture_to_texture(
            {"texture": self._render_texture, "mip_level": 0, "origin": (0, 0, 0)},
            {"texture": self._static_texture, "mip_level": 0, "origin": (0, 0, 0)},
            (config.pixel_width, config.pixel_height, 1),
        )
        self._device.queue.submit([encoder.finish()])

        self._has_static_frame = True

    def clear_screen(self) -> None:
        if self.window is not None:
            self.window.present()

    # ------------------------------------------------------------------
    # Skipping helpers
    # ------------------------------------------------------------------

    def update_skipping_status(self) -> None:
        """Check and update ``skip_animations`` for the current animation.

        Mirrors ``CairoRenderer.update_skipping_status`` and
        ``OpenGLRenderer.update_skipping_status`` so the WebGPU renderer
        honours the same configuration knobs:

        * ``file_writer.sections[-1].skip_animations`` — section-level skip
          (e.g. the section was marked skip via ``scene.next_section``).
        * ``config.save_last_frame`` — only the final frame matters; all
          intermediate animation frames can be skipped.
        * ``config.from_animation_number`` — skip animations before the given
          index (useful for scrubbing to a specific animation).
        * ``config.upto_animation_number`` — stop rendering after the given
          index and raise ``EndSceneEarlyException``.
        """
        # there is always at least one section → no out-of-bounds here
        if self.file_writer.sections[-1].skip_animations:
            self.skip_animations = True
        if config["save_last_frame"]:
            self.skip_animations = True
        if (
            config.from_animation_number > 0
            and self.num_plays < config.from_animation_number
        ):
            self.skip_animations = True
        if (
            config.upto_animation_number >= 0
            and self.num_plays > config.upto_animation_number
        ):
            self.skip_animations = True
            raise EndSceneEarlyException()

    def _should_save_last_frame(self) -> bool:
        if config["save_last_frame"]:
            return True
        if self.scene.interactive_mode:
            return False
        return self.num_plays == 0

    # ------------------------------------------------------------------
    # Background colour
    # ------------------------------------------------------------------

    @property
    def background_color(self):
        return self._background_color

    @background_color.setter
    def background_color(self, value) -> None:
        self._background_color = color_to_rgba(value, 1.0)

    def get_pixel_shape(self) -> tuple[int, int]:
        return (config.pixel_width, config.pixel_height)
