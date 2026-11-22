from __future__ import annotations

import contextlib
import dataclasses
import threading
import time
import typing
import weakref
from typing import TYPE_CHECKING, Any

import moderngl
import numpy as np
from moderngl import Framebuffer
from PIL import Image

from manim._config import config, logger
from manim.renderer.extract import extract
from manim.typing import Point3D
from manim.utils.color import color_to_rgba
from manim.utils.iterables import list_update

from ...camera import Camera, MultiCamera
from ...constants import *
from ...scene.scene_file_writer import SceneFileWriter
from .. import _view as _views
from .._execution import _RendererExecutionView
from ..protocol import CameraState, RendererCapabilities, RendererState
from ._drawing import DrawTarget, SnapshotDrawing
from .shader import shader_program_cache
from .window_settings import _WindowSettings

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from manim._config.render_session import RenderSessionSpec
    from manim.animation.animation import Animation
    from manim.mobject.mobject import Mobject, _AnimationBuilder
    from manim.scene.scene import Scene
    from manim.typing import (
        FloatRGBA,
        RGBAPixelArray,
    )
    from manim.utils.color.core import ParsableManimColor

    from .window import Window

__all__ = ["OpenGLRenderer"]

# Remember the active Manim renderer on each thread so captures can restore it.
_active_context = threading.local()


@dataclasses.dataclass(frozen=True)
class _SceneSettings:
    """What an OpenGL renderer draws a scene with, saved when it is bound to it."""

    pixel_size: tuple[int, int]
    wireframe: bool
    #: The preview window to open, if any.
    window: _WindowSettings | None


class OpenGLRenderer(_RendererExecutionView):
    """
    An OpenGL-based renderer.

    Attributes
    ----------
    animation_elapsed_time : float
        The elapsed time of the current animation.
    animation_start_time : float
        The start time of the current animation.
    animations_hashes : list[str | None]
        List of animation hashes for caching.
    anti_alias_width : float
        The width used for anti-aliasing in pixel units.
    camera : Camera
        The camera used for rendering.
    num_plays : float
        The number of animation plays executed.
    pressed_keys : set[int]
        Set of currently pressed key codes.
    skip_animations : bool
        Whether animations are currently being skipped.
    time : float
        The total elapsed time for the renderer.
    window : Window | None
        The window used for previewing, if any.
    """

    capabilities = RendererCapabilities(live_preview=True)

    def __init__(
        self,
        skip_animations: bool = False,
        camera_class: type[Camera] | None = None,
        camera: Camera | None = None,
    ) -> None:
        """Initializes the OpenGLRenderer.

        Parameters
        ----------
        skip_animations : bool, optional
            Whether to skip animations during rendering, by default False.
        camera_class
            The camera class to instantiate, by default :class:`.Camera`.
        camera
            A camera to use instead of instantiating ``camera_class``.
        """
        if camera is not None and camera_class is not None:
            raise ValueError("Pass either camera or camera_class, not both.")
        # Measured in pixel widths, used for vector graphics
        self.anti_alias_width = 1.5

        self._initialize_execution(skip_animations)
        self.animation_start_time = 0.0
        self.animation_elapsed_time = 0.0

        self.camera: Camera = camera or (camera_class or Camera)()
        self.perspective_uniforms: dict[str, Any] = {}
        self._camera_state: CameraState = self.camera._state()
        self.camera_view: _CameraView = _CameraView.of(self._camera_state)
        self._snapshot_drawing = SnapshotDrawing(self)
        self._camera_view_targets: dict[int, _ViewTarget] = {}
        self.pressed_keys: set[int] = set()
        self.window: Window | None = None
        self.background_color = config["background_color"]
        self._context: moderngl.Context | None = None
        self._frame_buffer_object: Framebuffer | None = None
        self._context_thread: int | None = None
        self._capturing_image = False
        # A finished scene whose final state is drawn on the first pixel read.
        self._final_scene: Scene | None = None
        self._keep_image_renderer = False
        self._kept_image_renderer: OpenGLRenderer | None = None
        self._state = RendererState.UNOPENED
        self._resources = contextlib.ExitStack()
        self._previous_renderer: weakref.ReferenceType[OpenGLRenderer] | None = None

    def init_scene(
        self,
        scene: Scene,
        session_spec: RenderSessionSpec,
    ) -> None:
        """
        Attach a scene and save its pixel dimensions and preview-window settings.

        :meth:`open` uses these settings to create the OpenGL context and
        framebuffer. The manager calls it before the scene's ``setup()`` method;
        an explicit request for GPU resources, such as :attr:`context`, opens
        them sooner.

        Parameters
        ----------
        scene : Scene
            The scene to be rendered
        """
        self._ensure_not_closed()
        if hasattr(self, "scene"):
            raise RuntimeError("This renderer is already bound to a Scene.")
        self.background_color = config["background_color"]
        self._bind(
            scene,
            _SceneSettings(
                pixel_size=(int(config.pixel_width), int(config.pixel_height)),
                wireframe=config.enable_wireframe,
                window=(
                    _WindowSettings.from_config(config)
                    if self.should_create_window(session_spec)
                    else None
                ),
            ),
        )

    def _bind(self, scene: Scene, settings: _SceneSettings) -> None:
        self.partial_movie_files: list[str | None] = []
        self.scene = scene
        self._settings = settings

    @classmethod
    def _image_renderer(cls, of: OpenGLRenderer) -> OpenGLRenderer:
        """An offscreen renderer drawing images of ``of``'s scene as ``of`` would:
        same camera, background and settings, but no window.
        """
        renderer = cls()
        renderer.camera = of.camera
        renderer.background_color = of.background_color
        renderer._bind(of.scene, dataclasses.replace(of._settings, window=None))
        return renderer

    @property
    def state(self) -> RendererState:
        """Where the renderer is in its life (read-only)."""
        return self._state

    def _ensure_not_closed(self) -> None:
        if not self._state.accepts_work:
            raise RuntimeError(f"The OpenGL renderer is {self._state.value}.")

    def open(self) -> None:
        """Create or activate the scene's OpenGL context.

        The first call creates the context, framebuffer, and preview window if
        requested. Further calls activate the same context. Use and close these
        resources on the thread that first opened them.
        """
        self._ensure_not_closed()
        if self._context is not None:
            if isinstance(self._context.mglo, moderngl.InvalidObject):
                raise RuntimeError("The OpenGL context was released externally.")
            if self.window is not None and self.window._window.context is None:
                raise RuntimeError("The OpenGL window was destroyed externally.")
            if self._context_thread != threading.get_ident():
                raise RuntimeError(
                    "OpenGL resources must be used on their owning thread."
                )
            self._activate_context()
            return
        if not hasattr(self, "scene"):
            raise RuntimeError("Bind an OpenGL scene before opening its resources.")
        previous = getattr(_active_context, "renderer", None)
        resources = contextlib.ExitStack()
        try:
            window = None
            if self._settings.window is not None:
                # local import: pyglet/screeninfo only needed for preview window
                from .window import Window

                window = Window(self, _settings=self._settings.window)
                resources.callback(window.close)
                # Closing the window also releases its context and default framebuffer.
                context = window.ctx
                frame = context.detect_framebuffer()
            else:
                try:
                    context = moderngl.create_context(standalone=True)
                except Exception:
                    context = moderngl.create_context(standalone=True, backend="egl")
                resources.callback(context.release)
                frame = self.get_frame_buffer_object(
                    context, 0, size=self._settings.pixel_size
                )
                for color in frame.color_attachments:
                    resources.callback(color.release)
                if frame.depth_attachment is not None:
                    resources.callback(frame.depth_attachment.release)
                resources.callback(frame.release)
                frame.use()
            context.enable(moderngl.BLEND)
            context.wireframe = self._settings.wireframe
            context.blend_func = (
                moderngl.SRC_ALPHA,
                moderngl.ONE_MINUS_SRC_ALPHA,
                moderngl.ONE,
                moderngl.ONE,
            )
        except BaseException:
            try:
                resources.close()
            except BaseException:
                logger.exception("Failed to roll back OpenGL initialization")
            try:
                self._restore_renderer(previous)
            except BaseException:
                logger.exception("Failed to restore the previous OpenGL context")
            raise
        # Store the resources after initialization succeeds.
        self.window = window
        self._context = context
        self._frame_buffer_object = frame
        self._context_thread = threading.get_ident()
        self._capturing_image = False
        self._resources = resources.pop_all()
        self._previous_renderer = previous
        _active_context.renderer = weakref.ref(self)
        self._state = RendererState.OPEN

    def _activate_context(self) -> None:
        if self.window is not None:
            self.window._activate_context()
        else:
            assert self._context is not None
            self._context.__enter__()
        _active_context.renderer = weakref.ref(self)

    @property
    def context(self) -> moderngl.Context:
        self.open()
        assert self._context is not None
        return self._context

    @property
    def frame_buffer_object(self) -> Framebuffer:
        self.open()
        assert self._frame_buffer_object is not None
        return self._frame_buffer_object

    @frame_buffer_object.setter
    def frame_buffer_object(self, frame: Framebuffer) -> None:
        self._ensure_not_closed()
        self._frame_buffer_object = frame

    def close(self) -> None:
        """Release GPU resources and close the preview window.

        Use :meth:`.Manager.close` for scene cleanup: it stops encoding jobs
        before closing the renderer. To render another scene, create a new
        renderer. :meth:`.Scene.get_image` can still draw with a temporary
        context.
        """
        if self._state is RendererState.CLOSED:
            return
        if self._capturing_image:
            raise RuntimeError("Cannot close OpenGL resources during image capture.")
        if self._context is not None and self._context_thread != threading.get_ident():
            raise RuntimeError(
                "OpenGL resources must be closed on their owning thread."
            )
        if self._context is None:
            self._state = RendererState.CLOSED
            return
        current = getattr(_active_context, "renderer", None)
        previous = (
            self._previous_renderer
            if current is not None and current() is self
            else current
        )
        # Release all resources before propagating a cleanup error or interruption.
        failures: list[BaseException] = []

        def release(callback: Callable[[], Any]) -> None:
            try:
                callback()
            except BaseException as error:
                failures.append(error)

        # Scene.interact may already have closed the window and released its GPU
        # objects. Deleting them again could affect a different active context.
        host_alive = (
            self._context is not None
            and not isinstance(self._context.mglo, moderngl.InvalidObject)
            and (self.window is None or self.window._window.context is not None)
        )
        if self._context is not None and host_alive:
            self._activate_context()
        self._state = RendererState.CLOSING
        for name, program in list(shader_program_cache.items()):
            if program.ctx is self._context:
                del shader_program_cache[name]
                if host_alive:
                    release(program.release)
        if host_alive:
            release(self._resources.close)
        else:
            self._resources.pop_all()
            if self.window is not None and self.window._window.context is not None:
                release(self.window.close)
        self.pressed_keys.clear()
        self._resources = contextlib.ExitStack()
        host_still_alive = (
            not isinstance(self._context.mglo, moderngl.InvalidObject)
            if self.window is None
            else self.window._window.context is not None
        )
        if failures and host_still_alive:
            # Keep the context reachable so close() can retry. Closing it releases
            # any remaining GPU objects; rendering stays disabled until then.
            self._resources.callback(
                self._context.release if self.window is None else self.window.close
            )
            self._state = RendererState.CLOSE_FAILED
        else:
            self._context = None
            self._frame_buffer_object = None
            self._context_thread = None
            self.window = None
            self._state = RendererState.CLOSED
        _active_context.renderer = None
        release(lambda: self._restore_renderer(previous))
        if failures:
            for secondary in failures[1:]:
                logger.error("Additional OpenGL cleanup failure", exc_info=secondary)
            raise failures[0]

    @staticmethod
    def _restore_renderer(
        reference: weakref.ReferenceType[OpenGLRenderer] | None,
    ) -> None:
        prior = reference() if reference is not None else None
        if (
            prior is not None
            and prior._state is RendererState.OPEN
            and prior._context is not None
            and not isinstance(prior._context.mglo, moderngl.InvalidObject)
            and (prior.window is None or prior.window._window.context is not None)
        ):
            prior.open()

    @property
    def file_writer(self) -> SceneFileWriter:
        """Return the scene manager's file writer, creating it if needed."""
        return self.scene._get_manager().file_writer

    def should_create_window(self, session_spec: RenderSessionSpec) -> bool:
        """Return whether the scene's saved settings request a live-preview window."""
        return session_spec.presentation.live_preview

    def get_pixel_shape(self) -> tuple[int, int] | None:
        """
        Return the pixel dimensions of the current framebuffer.

        Returns
        -------
        tuple[int, int] | None
            ``(width, height)`` in pixels, or ``None`` when no framebuffer is open.
        """
        frame_buffer = self._frame_buffer_object
        if frame_buffer is None:
            return None
        _, _, pixel_width, pixel_height = frame_buffer.viewport
        return pixel_width, pixel_height

    def refresh_perspective_uniforms(self, camera: Camera) -> None:
        """Update the camera uniforms from ``camera``.

        The shaders rotate points about the frame center, divide by the frame
        shape and apply a perspective factor ``focal_distance / (focal_distance - z)``,
        which is what :meth:`.ThreeDCamera.project_points` does on the CPU. A
        camera without rotation draws orthographically.

        Parameters
        ----------
        camera
            The camera whose view to draw.

        Raises
        ------
        ValueError
            If the renderer's pixel shape is not available.
        """
        self.open()
        pixel_shape = self.get_pixel_shape()
        if pixel_shape is None:
            msg = "Pixel shape is None, cannot refresh perspective uniforms."
            raise ValueError(msg)
        _, pixel_height = pixel_shape
        self._camera_state = camera._state()
        view = self.camera_view = _CameraView.of(self._camera_state)
        self.perspective_uniforms = self.camera_uniforms(view, pixel_height)

    def camera_uniforms(self, view: _CameraView, pixel_height: int) -> dict[str, Any]:
        """The shaders' camera uniforms for a view drawn ``pixel_height`` pixels high."""
        _, frame_height = view.frame_shape
        return {
            "frame_shape": view.frame_shape,
            # Antialiasing width, measured in pixels, converted to scene units.
            "anti_alias_width": self.anti_alias_width / (pixel_height / frame_height),
            "camera_center": tuple(view.center),
            "camera_rotation": tuple(view.rotation.T.flatten()),
            "focal_distance": view.focal_distance,
        }

    def _draw_snapshots(self, scene: Scene) -> None:
        """Draw the scene's mobjects from render snapshots, camera views first."""
        mobjects = list_update(scene.mobjects, scene.foreground_mobjects)
        views = self._draw_camera_views(mobjects)
        nodes = extract(mobjects, use_z_index=self.camera.use_z_index)
        pixel_shape = self.get_pixel_shape()
        assert pixel_shape is not None
        target = DrawTarget(
            self.frame_buffer_object,
            pixel_shape,
            self._camera_state,
            self.camera_view,
            self.perspective_uniforms,
        )
        self._snapshot_drawing.draw(
            _views.sort_for_display(self._camera_state, nodes), target, views
        )

    def _draw_camera_views(self, mobjects: list[Mobject]) -> dict[int, Any]:
        """Draw what each camera view of a :class:`.MultiCamera` sees into a texture.

        Returns the textures by the uid of their :class:`.ImageMobjectFromCamera`.
        """
        if not isinstance(self.camera, MultiCamera):
            return {}
        pixel_shape = self.get_pixel_shape()
        assert pixel_shape is not None
        textures = {}
        for image_mobject in self.camera.image_mobjects_from_cameras:
            size = (
                max(
                    1,
                    int(pixel_shape[0] * image_mobject.width / self.camera.frame_width),
                ),
                max(
                    1,
                    int(
                        pixel_shape[1] * image_mobject.height / self.camera.frame_height
                    ),
                ),
            )
            view_target = self._camera_view_targets.get(image_mobject.uid)
            if view_target is None or view_target.size != size:
                if view_target is not None:
                    view_target.release()
                view_target = _ViewTarget(self, size)
                self._camera_view_targets[image_mobject.uid] = view_target
            camera = image_mobject.camera
            state = camera._state()
            view = _CameraView.of(state)
            target = DrawTarget(
                view_target.framebuffer,
                size,
                state,
                view,
                self.camera_uniforms(view, size[1]),
            )
            view_target.framebuffer.use()
            view_target.framebuffer.clear(
                *camera.background_color.to_rgb(), camera.background_opacity
            )
            nodes = extract(
                mobjects,
                use_z_index=camera.use_z_index,
                excluded_mobjects=[image_mobject],
            )
            view_target.drawing.draw(_views.sort_for_display(state, nodes), target)
            textures[image_mobject.uid] = view_target.texture
        for uid in self._camera_view_targets.keys() - textures.keys():
            self._camera_view_targets.pop(uid).release()
        return textures

    def update_skipping_status(self) -> None:
        """Ask the scene's manager whether this play should be fast-forwarded.

        Raises
        ------
        EndSceneEarlyException
            If the number of played animations exceeds the configured upper bound.
        """
        self.scene._get_manager()._update_skipping_status()

    def play(
        self,
        scene: Scene,
        *animations: Animation | Mobject | _AnimationBuilder,
        **kwargs: Any,
    ) -> None:
        """Delegate animation playback to the scene's manager.

        Prefer :meth:`.Scene.play` for new code; it also handles subcaptions and
        calls made from another thread during OpenGL interaction.

        Parameters
        ----------
        scene
            The scene in which to play the animations.
        animations
            The animations, mobjects, or animation builders to play.
        kwargs
            Animation options such as ``run_time`` and ``rate_func``.
        """
        scene._get_manager().play(*animations, **kwargs)

    def _animation_cache_identity(self, scene: Scene) -> tuple[str, Any]:
        return "opengl", {
            "background_color": self.background_color,
            "anti_alias_width": self.anti_alias_width,
        }

    def _start_animation(self) -> None:
        self.open()
        self.animation_start_time = time.time()

    def _prepare_animation(self, scene: Scene) -> None:
        pass

    def _present_frozen_frame(self, scene: Scene, duration: float) -> None:
        if self.window is not None:
            self.window.swap_buffers()
            while time.time() - self.animation_start_time < duration:
                pass
        self.animation_elapsed_time = duration

    def clear_screen(self) -> None:
        """
        Clears the current frame buffer and updates the display window
        accordingly.

        The screen is cleared using the background color specified
        in the renderer.
        """
        self.frame_buffer_object.clear(*self.background_color)
        if self.window is None:
            return
        self.window.swap_buffers()

    def render(
        self, scene: Scene, frame_offset: float, moving_mobjects: list[Mobject]
    ) -> None:
        """Draw a single frame of the scene's current state using OpenGL.

        Parameters
        ----------
        scene
            The scene to draw.
        frame_offset
            Animation-relative time in seconds. Accepted for the common renderer
            interface; drawing itself does not use this value.
        moving_mobjects
            Accepted for the common renderer interface. OpenGL draws the full
            scene rather than using Cairo's moving/static split.

        Notes
        -----
        This method draws into the framebuffer. The manager reads any pixels
        needed for output and displays the preview window separately. Drawing
        alone neither writes frames nor advances ``Scene.time``.
        """
        self.update_frame(scene)

    def _present_frame(self, scene: Scene, frame_offset: float) -> None:
        """Refresh the preview window until wall-clock time reaches the frame offset."""
        if self.window is not None:
            self.window.swap_buffers()
            while self.animation_elapsed_time < frame_offset:
                self.update_frame(scene)
                self.window.swap_buffers()

    def update_frame(self, scene: Scene) -> None:
        """
        Update and render the current frame for the given scene.

        Performs the following steps:
        1. Clear the frame buffer with the background color.
        2. Refresh camera perspective uniforms for rendering.
        3. Draw render snapshots of the scene's mobjects.
        4. Iterate through all mesh objects in the scene, setting their uniforms and rendering them.
        5. Update the wall-clock time used to pace the preview window.

        Parameters
        ----------
        scene : Scene
            The scene to render the frame for.
        """
        self._final_scene = None
        self._draw_scene(scene)
        self.animation_elapsed_time = time.time() - self.animation_start_time

    def _draw_scene(self, scene: Scene) -> None:
        """Draw the current objects without updating animation timing."""
        self.frame_buffer_object.clear(*self.background_color)

        self.refresh_perspective_uniforms(self.camera)

        self._draw_snapshots(scene)

    @contextlib.contextmanager
    def _scene_images(self) -> Iterator[None]:
        """Keep the temporary renderer of :meth:`_get_scene_image` for many images.

        Creating an OpenGL context for each image costs far more than drawing it,
        so evaluation, which draws every requested frame this way, keeps one.
        """
        self._keep_image_renderer = True
        try:
            yield
        finally:
            self._keep_image_renderer = False
            kept, self._kept_image_renderer = self._kept_image_renderer, None
            if kept is not None:
                kept.close()

    def _get_scene_image(self, scene: Scene) -> Image.Image:
        """Draw a snapshot separately from the live-preview framebuffer.

        An existing context is used on its rendering thread. Otherwise, the scene
        is drawn with a temporary context and renderer.
        """
        if (
            self._context_thread is not None
            and threading.get_ident() != self._context_thread
        ):
            raise RuntimeError(
                "OpenGL scene images must be requested on the render thread."
            )
        if self._capturing_image:
            raise RuntimeError("Recursive OpenGL scene image capture is not supported.")
        if self._context is None:
            kept = self._kept_image_renderer
            temporary = (
                kept if kept is not None else OpenGLRenderer._image_renderer(self)
            )
            self._capturing_image = True
            try:
                try:
                    temporary._draw_scene(scene)
                    image = temporary.get_image()
                except BaseException:
                    self._kept_image_renderer = None
                    try:
                        temporary.close()
                    except BaseException:
                        logger.exception(
                            "Failed to close the temporary OpenGL image scope"
                        )
                    raise
                if self._keep_image_renderer:
                    self._kept_image_renderer = temporary
                else:
                    temporary.close()
                return image
            finally:
                self._capturing_image = False
        target = self.frame_buffer_object
        bound = self.context.fbo
        viewport = self.context.viewport
        size = target.size
        self._capturing_image = True
        try:
            with contextlib.ExitStack() as resources:
                color = self.context.texture(size, components=4)
                resources.callback(color.release)
                depth = self.context.depth_renderbuffer(size)
                resources.callback(depth.release)
                frame = self.context.framebuffer(color, depth)
                resources.callback(frame.release)
                try:
                    self.frame_buffer_object = frame
                    frame.use()
                    self._draw_scene(scene)
                    return Image.fromarray(self.get_frame())
                finally:
                    self.frame_buffer_object = target
                    (target if bound is None else bound).use()
                    self.context.viewport = viewport
        finally:
            self._capturing_image = False

    def scene_finished(self, scene: Scene) -> None:
        """Finalize configured output for the scene.

        Parameters
        ----------
        scene
            The scene that has finished rendering.
        """
        output = self.file_writer.output_spec
        if self.num_plays > 0 and (output.is_video or output.is_image_sequence):
            self.file_writer.finish()
        # Skipped and cached plays do not draw, so the framebuffer may not show
        # the final state. A preview window shows it; offscreen, it is drawn
        # when the pixels are read, not for every scene.
        if self.window is not None:
            self.update_frame(scene)
        else:
            self._final_scene = scene

        if self.should_save_last_frame():
            self.file_writer.save_image(self.get_frame())

    def should_save_last_frame(self) -> bool:
        """
        Determine whether the last frame of the scene should be saved.

        This is true for explicit last-frame PNG output and for automatic video
        output when the scene has no play calls. Interactive scenes do not use
        the automatic fallback.
        """
        output = self.file_writer.output_spec
        if output.is_still:
            return True
        if self.scene.interactive_mode:
            return False
        return self.num_plays == 0 and output.fallback_to_still

    def get_image(self) -> Image.Image:
        """
        Get the current OpenGL frame buffer as a PIL Image.

        Returns
        -------
        Image.Image
            The image representation of the current frame buffer.

        Raises
        ------
        ValueError
            If the pixel shape cannot be determined.

        Notes
        -----
        The image is constructed from raw RGBA buffer data, with the
        origin at the bottom-left.
        """
        raw_buffer_data = self.get_raw_frame_buffer_object_data()
        pixel_shape = self.get_pixel_shape()
        if pixel_shape is None:
            msg = "Pixel shape is None, cannot get image."
            raise ValueError(msg)

        image = Image.frombytes(
            "RGBA",  # mode (rgb, a for alpha (transparency)))
            pixel_shape,  # size
            raw_buffer_data,  # data
            "raw",  # decoder_name
            # *args for the decoder
            "RGBA",  # raw mode
            0,  # stride (O = no extra padding)
            -1,  # orientation (-1 = bottom to top, 1 = top to bottom)
        )
        return image

    def save_static_frame_data(
        self, scene: Scene, static_mobjects: Iterable[Mobject]
    ) -> None:
        self._ensure_not_closed()

    def get_frame_buffer_object(
        self,
        context: moderngl.Context,
        samples: int = 0,
        *,
        size: tuple[int, int] | None = None,
    ) -> Framebuffer:
        """
        Creates and returns a framebuffer object configured with color
        and depth attachments.

        Parameters
        ----------
        context : moderngl.Context
            The ModernGL context used to create the framebuffer and
            its attachments.
        samples : int, optional
            The number of samples for multisample anti-aliasing (MSAA) [1]_.
            Default is 0 (no MSAA).

        Returns
        -------
        Framebuffer
            A framebuffer object with a color texture attachment and
            a depth renderbuffer attachment, both sized according to
            the current configuration's pixel width and height.

        Notes
        -----
        Framebuffer's color attachment is supposed RGBA.
        Pixel dimensions are taken from the global config of Manim.

        References
        ----------
        .. [1] Wikipedia, "Multisample anti-aliasing",
               https://en.wikipedia.org/wiki/Multisample_anti-aliasing
        """
        pixel_width, pixel_height = (
            (config["pixel_width"], config["pixel_height"]) if size is None else size
        )
        num_channels = 4
        resources = contextlib.ExitStack()
        try:
            color = context.texture(
                (pixel_width, pixel_height), components=num_channels, samples=samples
            )
            resources.callback(color.release)
            depth = context.depth_renderbuffer(
                (pixel_width, pixel_height), samples=samples
            )
            resources.callback(depth.release)
            frame = context.framebuffer(color_attachments=color, depth_attachment=depth)
        except BaseException:
            try:
                resources.close()
            except BaseException:
                logger.exception("Failed to roll back OpenGL framebuffer allocation")
            raise
        resources.pop_all()
        return frame

    def get_raw_frame_buffer_object_data(self, dtype: str = "f1") -> bytes:
        """
        Get the raw data from the current frame buffer object as bytes.

        This method reads the pixel data from the frame buffer object using the specified data type.
        The data is read with 4 color channels (typically RGBA).

        Args:
            dtype (str, optional): The data type to use when reading the buffer.
            Defaults to "f1" (i.e., float with 1 byte).

        Returns:
            bytes: The raw pixel data from the frame buffer object.
        """
        # Copy blocks from the fbo_msaa to the drawn fbo using Blit
        # pw, ph = self.get_pixel_shape()
        # gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.fbo_msaa.glo)
        # gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.fbo.glo)
        # gl.glBlitFramebuffer(
        #     0, 0, pw, ph, 0, 0, pw, ph, gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR
        # )
        if self._final_scene is not None:
            scene, self._final_scene = self._final_scene, None
            self.update_frame(scene)
        num_channels = 4
        ret: bytes = self.frame_buffer_object.read(
            viewport=self.frame_buffer_object.viewport,
            components=num_channels,
            dtype=dtype,
        )
        return ret

    def get_frame(self) -> RGBAPixelArray:
        """
        Get the current frame buffer as a Numpy array of RGBA pixel values.

        Returns
        -------
        RGBAPixelArray
            A Numpy array of shape (height, width, 4) containing the
            RGBA pixel data of the current frame, with dtype uint8.

        Raises
        ------
        ValueError
            If the pixel shape cannot be determined.
        """
        # get current pixel values as numpy data in order to test output
        raw = self.get_raw_frame_buffer_object_data(dtype="f1")
        pixel_shape = self.get_pixel_shape()
        if pixel_shape is None:
            msg = "Pixel shape is None, cannot get frame."
            raise ValueError(msg)

        result_dimensions = (pixel_shape[1], pixel_shape[0], 4)
        np_buf = np.frombuffer(raw, dtype="uint8").reshape(result_dimensions)
        return np.flipud(np_buf).copy()

    # Returns offset from the bottom left corner in pixels.
    # top_left flag should be set to True when using a GUI framework
    # where the (0,0) is at the top left: e.g. PySide6
    def pixel_coords_to_space_coords(
        self, px: float, py: float, relative: bool = False, top_left: bool = False
    ) -> Point3D:
        """
        Converts pixel coordinates to space (scene) coordinates.

        top_left flag should be set to True when using a GUI framework
        where the (0,0) is at the top left: e.g. PySide6.

        Parameters
        ----------
        px : float
            The x-coordinate in pixel space.
        py : float
            The y-coordinate in pixel space.
        relative : bool, optional
            If True, returns coordinates relative to the frame (normalized to [-1, 1]).
            If False, returns absolute space coordinates. Default is False.
        top_left : bool, optional
            If True, treats the origin (0, 0) as the top-left corner of the pixel space.
            If False, treats the origin as the bottom-left. Default is False.

        Returns
        -------
        Point3D
            The corresponding coordinates in space as a NumPy array of shape (3,).

        Notes
        -----
        If the pixel shape is not available, returns the origin [0, 0, 0].
        """
        pixel_shape = self.get_pixel_shape()
        if pixel_shape is None:
            return typing.cast(Point3D, np.array([0.0, 0.0, 0.0]))
        pixel_width, pixel_height = pixel_shape
        view = _CameraView.of(self.camera._state())
        frame_height = view.frame_shape[1]
        frame_center = view.center
        if relative:
            # relative -> just normalize to [-1, 1]
            return 2 * np.array([px / pixel_width, py / pixel_height, 0])

        scale = frame_height / pixel_height
        y_direction = -1 if top_left else 1

        return typing.cast(
            Point3D,
            frame_center
            + scale
            * np.array(
                [(px - pixel_width / 2), y_direction * (py - pixel_height / 2), 0.0]
            ),
        )

    @property
    def background_color(self) -> FloatRGBA:
        """The background color of the renderer (RGBA format)."""
        return self._background_color

    @background_color.setter
    def background_color(self, value: ParsableManimColor) -> None:
        self._background_color = color_to_rgba(value, 1.0)


#: Focal distance standing in for "no perspective" (a camera without rotation).
_ORTHOGRAPHIC_FOCAL_DISTANCE = 1e6


def _eye(center: Point3D, rotation: np.ndarray, focal_distance: float) -> np.ndarray:
    """The camera position in scene coordinates: at the focal distance from the
    frame center, on the view axis.
    """
    return np.asarray(center, dtype=float) + np.asarray(rotation).T @ np.array(
        [0.0, 0.0, focal_distance]
    )


class _CameraView(typing.NamedTuple):
    """What the shaders need to know about a camera."""

    center: Point3D
    #: Rotation applied to ``point - center``, as in ``ThreeDCamera.project_points``.
    rotation: np.ndarray
    #: Frame width and height, divided by the zoom factor.
    frame_shape: tuple[float, float]
    focal_distance: float
    #: The camera's light, in scene coordinates; ``None`` without one.
    light_source: Point3D | None
    #: Where the scene is seen from, in scene coordinates.
    eye: Point3D

    @classmethod
    def of(cls, state: CameraState) -> _CameraView:
        zoom = state.zoom
        focal_distance = state.focal_distance
        if focal_distance is None:
            focal_distance = _ORTHOGRAPHIC_FOCAL_DISTANCE
        return cls(
            center=np.asarray(state.center, dtype=float),
            rotation=state.rotation,
            frame_shape=(state.frame_width / zoom, state.frame_height / zoom),
            focal_distance=focal_distance,
            light_source=state.light,
            eye=_eye(state.center, state.rotation, focal_distance),
        )


class _ViewTarget:
    """An offscreen target for one camera view, with its own drawing state."""

    def __init__(self, renderer: OpenGLRenderer, size: tuple[int, int]) -> None:
        context = renderer.context
        self.size = size
        self.texture = context.texture(size, components=4)
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.framebuffer = context.framebuffer(color_attachments=[self.texture])
        self.drawing = SnapshotDrawing(renderer)
        renderer._resources.callback(self.release)

    def release(self) -> None:
        """Release the GPU objects; safe to call more than once."""
        self.drawing.release()
        for resource in (self.framebuffer, self.texture):
            if not isinstance(resource.mglo, moderngl.InvalidObject):
                resource.release()
