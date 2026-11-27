from __future__ import annotations

import contextlib
import threading
import typing
import weakref
from typing import TYPE_CHECKING, Any

import moderngl
import numpy as np
from moderngl import Framebuffer

from manim._config import logger
from manim.typing import Point3D

from ...constants import *
from .. import _view as _views
from ..protocol import (
    CapabilityError,
    PreviewRequest,
    RenderConfig,
    RendererState,
)
from ._drawing import DrawTarget, SnapshotDrawing
from .shader import shader_program_cache
from .window_settings import _WindowSettings

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from manim.typing import (
        RGBAPixelArray,
    )

    from ..protocol import CameraState, Frame
    from ..snapshot import RNode
    from .window import Window

__all__ = ["OpenGLRenderer"]

# Remember the active Manim renderer on each thread so captures can restore it.
_active_context = threading.local()


class OpenGLRenderer:
    """A renderer drawing with OpenGL, which can show frames in a preview window.

    Frames are drawn offscreen at the size their :class:`.RenderConfig` asks for. A
    renderer given a preview request has a :attr:`preview` handle, whose window
    shows frames scaled to it.

    Parameters
    ----------
    wireframe
        Draw the edges of triangles only, for debugging.
    anti_alias_width
        The width of the antialiased edge of shapes, in pixels.

    Attributes
    ----------
    anti_alias_width : float
        The width used for anti-aliasing in pixel units.
    window : Window | None
        The native preview window while it is open.
    """

    def __init__(
        self,
        *,
        wireframe: bool = False,
        anti_alias_width: float = 1.5,
    ) -> None:
        self.wireframe = wireframe
        # Measured in pixel widths, used for vector graphics
        self.anti_alias_width = anti_alias_width

        self.perspective_uniforms: dict[str, Any] = {}
        # Set for each frame by _set_view.
        self.camera_view: _CameraView
        self._camera_state: CameraState
        self._snapshot_drawing = SnapshotDrawing(self)
        self._camera_view_targets: dict[int, _ViewTarget] = {}
        #: The preview handle; ``None`` if no preview was requested.
        self.preview: _WindowPreview | None = None
        self.window: Window | None = None
        self._context: moderngl.Context | None = None
        self._frame_buffer_object: Framebuffer | None = None
        self._context_thread: int | None = None
        #: The number of frames drawn; a frame can be read until the next one.
        self._frames_drawn = 0
        self._state = RendererState.UNOPENED
        self._resources = contextlib.ExitStack()
        self._previous_renderer: weakref.ReferenceType[OpenGLRenderer] | None = None

    def _request_preview(
        self, request: PreviewRequest | None, settings: _WindowSettings | None
    ) -> None:
        """Give the renderer a preview window to open, with its settings (for
        :meth:`.OpenGL.create`).
        """
        self._ensure_not_closed()
        if request is None or settings is None:
            self.preview = None
        else:
            self.preview = _WindowPreview(self, request, settings)

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
        previous = getattr(_active_context, "renderer", None)
        resources = contextlib.ExitStack()
        try:
            window = None
            if self.preview is not None:
                # local import: pyglet/screeninfo only needed for preview window
                from .window import Window

                preview = self.preview
                window = Window(
                    preview.request.input,
                    to_scene=preview._scene_point,
                    _settings=preview.settings,
                )
                resources.callback(window.close)
                # Closing the window also releases its context.
                context = window.ctx
            else:
                try:
                    context = moderngl.create_context(standalone=True)
                except Exception:
                    context = moderngl.create_context(standalone=True, backend="egl")
                resources.callback(context.release)
            context.enable(moderngl.BLEND)
            context.wireframe = self.wireframe
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
        self._context_thread = threading.get_ident()
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
        """The offscreen framebuffer of the last frame drawn, which a preview
        window shows. It is created by the first :meth:`render`.
        """
        self.open()
        if self._frame_buffer_object is None:
            raise RuntimeError("The OpenGL renderer has not drawn a frame yet.")
        return self._frame_buffer_object

    def _size_frame(self, size: tuple[int, int]) -> None:
        """Make the offscreen framebuffer ``size`` pixels, as the frame to draw is."""
        frame = self._frame_buffer_object
        if frame is not None and frame.size == size:
            return
        replacement = self._make_framebuffer(self.context, 0, size=size)
        self._resources.callback(_release_framebuffer, replacement)
        if frame is not None:
            _release_framebuffer(frame)
        self._frame_buffer_object = replacement

    def close(self) -> None:
        """Release GPU resources and close the preview window.

        Use :meth:`.Manager.close` for scene cleanup: it stops encoding jobs
        before closing the renderer. To render another scene, create a new
        renderer. :meth:`.Scene.get_image` can still draw with a temporary
        context.
        """
        if self._state is RendererState.CLOSED:
            return
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
        if self.preview is not None:
            self.preview._pressed_keys.clear()
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

    def _set_view(self, config: RenderConfig) -> None:
        """Set the camera uniforms for the frame of ``config``.

        The shaders rotate points about the frame center, divide by the frame
        shape and apply a perspective factor ``focal_distance / (focal_distance - z)``,
        which is what :meth:`.ThreeDCamera.project_points` does on the CPU. A
        camera without rotation draws orthographically.
        """
        self._camera_state = config.camera
        view = self.camera_view = _CameraView.of(config.camera)
        self.perspective_uniforms = self.camera_uniforms(view, config.pixel_height)

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

    def _frame_target(self, config: RenderConfig) -> DrawTarget:
        return DrawTarget(
            self.frame_buffer_object,
            (config.pixel_width, config.pixel_height),
            self._camera_state,
            self.camera_view,
            self.perspective_uniforms,
            config.reference_pixel_size,
            config.lights,
            config.ambient,
        )

    def render(self, nodes: Sequence[RNode], config: RenderConfig, /) -> _GLFrame:
        """Draw ``nodes`` as ``config`` sets them up; return the frame.

        The picture stays in the renderer's framebuffer, which a preview window
        shows; it is read back only when its pixels are asked for.
        """
        self._draw(nodes, config)
        self._frames_drawn += 1
        return _GLFrame(
            self, self._frames_drawn, (config.pixel_width, config.pixel_height)
        )

    def _draw(self, nodes: Sequence[RNode], config: RenderConfig) -> None:
        """Draw a frame into the framebuffer, camera views first."""
        for frame in (config, *(view.config for view in config.camera_views)):
            if frame.background_image is not None:
                raise CapabilityError(
                    "The OpenGL renderer cannot draw a background image "
                    f"({frame.background_image!r}); use the Cairo renderer, or a "
                    "background color."
                )
        self.open()
        self._size_frame((config.pixel_width, config.pixel_height))
        self._set_view(config)
        views = self._draw_camera_views(config)
        self.frame_buffer_object.use()
        background = _background_rgba(config)
        self.frame_buffer_object.clear(*background)
        self._snapshot_drawing.draw(
            _views.sort_for_display(config.camera, list(nodes)),
            self._frame_target(config),
            views,
            background=background,
        )

    def _draw_camera_views(self, config: RenderConfig) -> dict[Any, Any]:
        """Draw each camera view into a texture; return them by the uid of the
        view's node, and by ``("camera", camera_id)`` for copies of a view.
        """
        textures: dict[Any, Any] = {}
        drawn = set()
        for view in config.camera_views:
            view_config = view.config
            size = (view_config.pixel_width, view_config.pixel_height)
            view_target = self._camera_view_targets.get(view.uid)
            if view_target is None or view_target.size != size:
                if view_target is not None:
                    view_target.release()
                view_target = _ViewTarget(self, size)
                self._camera_view_targets[view.uid] = view_target
            state = view_config.camera
            camera_view = _CameraView.of(state)
            target = DrawTarget(
                view_target.framebuffer,
                size,
                state,
                camera_view,
                self.camera_uniforms(camera_view, size[1]),
                view_config.reference_pixel_size,
                view_config.lights,
                view_config.ambient,
            )
            view_target.framebuffer.use()
            background = _background_rgba(view_config)
            view_target.framebuffer.clear(*background)
            view_target.drawing.draw(
                _views.sort_for_display(state, list(view.nodes)),
                target,
                background=background,
            )
            textures[view.uid] = textures[("camera", view.camera_id)] = (
                view_target.texture
            )
            drawn.add(view.uid)
        for uid in self._camera_view_targets.keys() - drawn:
            self._camera_view_targets.pop(uid).release()
        return textures

    @staticmethod
    def _make_framebuffer(
        context: moderngl.Context,
        samples: int = 0,
        *,
        size: tuple[int, int],
    ) -> Framebuffer:
        """A framebuffer of ``size`` pixels with an RGBA texture and a depth
        renderbuffer, multisampled with ``samples`` samples (0: not).
        """
        pixel_width, pixel_height = size
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

    def _read_pixels(self) -> RGBAPixelArray:
        """The framebuffer's pixels as RGBA, row zero at the top; a new array."""
        frame = self.frame_buffer_object
        width, height = frame.size
        raw = frame.read(viewport=(0, 0, width, height), components=4, dtype="f1")
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
        return np.flipud(pixels).copy()


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


class _WindowPreview:
    """The preview handle of an OpenGL renderer: the window opens with the
    renderer's context (on :meth:`show`, or with the first frame) and closes with
    the renderer.
    """

    def __init__(
        self,
        renderer: OpenGLRenderer,
        request: PreviewRequest,
        settings: _WindowSettings,
    ) -> None:
        self._renderer = renderer
        self.request = request
        self.settings = settings
        self._pressed_keys: set[int] = set()

    @property
    def is_closing(self) -> bool:
        window = self._renderer.window
        return window is not None and bool(window.is_closing)

    @property
    def pressed_keys(self) -> set[int]:
        window = self._renderer.window
        return window.pressed_keys if window is not None else self._pressed_keys

    def show(self) -> None:
        self._renderer.open()

    def present(self, frame: Frame) -> None:
        """Show ``frame``, the last frame the renderer drew, in the window.

        The window's context is made current first: another renderer's drawing,
        such as an image's, may have left its own context current.
        """
        renderer = self._renderer
        if not isinstance(frame, _GLFrame) or frame._renderer is not renderer:
            raise ValueError("A preview shows frames of its own renderer.")
        if frame._index != renderer._frames_drawn:
            raise RuntimeError("The renderer drew another frame since this one.")
        renderer.open()
        window = renderer.window
        assert window is not None
        assert renderer._frame_buffer_object is not None
        window.show(renderer._frame_buffer_object)

    def _scene_point(self, x: float, y: float, relative: bool) -> Point3D:
        """The scene point at a window position given as fractions of its width and
        height from the bottom left; with ``relative``, the scene offset of a move
        by those fractions.
        """
        view = _CameraView.of(self.request.view())
        width, height = view.frame_shape
        offset = np.array([x * width, y * height, 0.0])
        if relative:
            return typing.cast(Point3D, offset)
        return typing.cast(
            Point3D, view.center + offset - np.array([width / 2, height / 2, 0.0])
        )


class _GLFrame:
    """A frame of an OpenGL renderer: read back from its framebuffer the first
    time its pixels are asked for, until the renderer draws the next frame.
    """

    __slots__ = ("_index", "_pixels", "_renderer", "size")

    def __init__(
        self, renderer: OpenGLRenderer, index: int, size: tuple[int, int]
    ) -> None:
        self._renderer = renderer
        self._index = index
        self._pixels: RGBAPixelArray | None = None
        self.size = size

    def pixels(self) -> RGBAPixelArray:
        if self._pixels is None:
            renderer = self._renderer
            if renderer._frames_drawn != self._index or not renderer.state.accepts_work:
                raise RuntimeError(
                    "The OpenGL renderer drew another frame since this one, or "
                    "closed: read the pixels of a frame before drawing the next."
                )
            self._pixels = renderer._read_pixels()
        return self._pixels


def _release_framebuffer(frame: Framebuffer) -> None:
    """Release ``frame`` and its attachments; safe to call more than once."""
    if isinstance(frame.mglo, moderngl.InvalidObject):
        return
    # A released framebuffer forgets its attachments: release them first.
    for attachment in (*frame.color_attachments, frame.depth_attachment):
        if attachment is not None and not isinstance(
            attachment.mglo, moderngl.InvalidObject
        ):
            attachment.release()
    frame.release()


def _background_rgba(config: RenderConfig) -> tuple[float, float, float, float]:
    """The background of a frame, as a framebuffer is cleared to it."""
    r, g, b = config.background_color.to_rgb()
    return float(r), float(g), float(b), float(config.background_opacity)
