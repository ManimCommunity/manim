from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Self

from ..._config import logger
from ..._run.frames import frame_count
from ...scene.scene_file_writer import SceneFileWriter
from .._execution import _RendererExecutionView
from .._reuse import PrefixMemo, node_key, view_key
from .._view import sort_for_display
from ..protocol import PixelFrame, RenderConfig, RendererCapabilities, RendererState
from ..snapshot import RMesh
from .mesh_faces import mesh_faces
from .rendering import _CairoDrawingContext, reusable_boundary
from .target import _CairoRasterSettings, _CairoRenderTarget, _ImageFiles

if TYPE_CHECKING:
    from manim._config.render_session import RenderSessionSpec
    from manim.animation.animation import Animation
    from manim.mobject.mobject import Mobject, _AnimationBuilder
    from manim.scene.scene import Scene

    from ...typing import RGBAPixelArray
    from ..snapshot import RCameraView, RNode

__all__ = ["CairoRenderer"]


class CairoRenderer(_RendererExecutionView):
    """A renderer using Cairo.

    The renderer draws frames, including the pictures of camera views, into image
    buffers of the size each frame's :class:`.RenderConfig` asks for, allocated
    when it first draws one. It keeps the drawing of what did not change between
    frames.
    """

    capabilities = RendererCapabilities(live_preview=False)

    def __init__(self, skip_animations: bool = False) -> None:
        self._initialize_execution(skip_animations)
        # Image files read for this renderer, shared by all its targets.
        self._image_files = _ImageFiles()
        self._target: _CairoRenderTarget | None = None
        self._sub_targets: dict[int, _CairoRenderTarget] = {}
        # The drawing of the leading nodes of a frame that did not change, and
        # their keys.
        self._memo = PrefixMemo()
        self._saved_pixels: RGBAPixelArray | None = None
        self._state = RendererState.UNOPENED

    @property
    def state(self) -> RendererState:
        """Where the renderer is in its life (read-only)."""
        return self._state

    def _ensure_open(self) -> None:
        """Check that the renderer is not closed."""
        if not self._state.accepts_work:
            raise RuntimeError(f"The Cairo renderer is {self._state.value}.")

    def init_scene(
        self,
        scene: Scene,
        session_spec: RenderSessionSpec,
    ) -> None:
        self._ensure_open()
        if hasattr(self, "_scene"):
            raise RuntimeError("This renderer is already bound to a Scene.")
        self._scene = scene

    @property
    def file_writer(self) -> SceneFileWriter:
        """Return the scene manager's file writer, creating it if needed."""
        return self._scene._get_manager().file_writer

    def play(
        self,
        scene: Scene,
        *args: Animation | Mobject | _AnimationBuilder,
        **kwargs: Any,
    ) -> None:
        """Delegate playback to the scene's manager; prefer Scene.play for new code."""
        scene._get_manager().play(*args, **kwargs)

    def _animation_cache_identity(self, scene: Scene) -> tuple[str, Any]:
        return "cairo", ()

    def _start_animation(self) -> None:
        self._ensure_open()

    def _present_frame(self, scene: Scene, frame_offset: float) -> None:
        pass

    def _present_frozen_frame(self, scene: Scene, duration: float) -> None:
        pass

    def render(self, nodes: Sequence[RNode], config: RenderConfig, /) -> PixelFrame:
        """Draw ``nodes`` as ``config`` sets them up; return the frame, with a copy
        of the pixels that belongs to the caller.

        Camera views are drawn first, each into its own image, and shown on their
        nodes. The drawing of the leading nodes that did not change since the
        previous frame is copied instead of drawn again (see
        :mod:`manim.renderer._reuse`).
        """
        self._ensure_open()
        target = self._target_for(config)
        pictures = self._draw_camera_views(config)

        def resolve_image(node: RNode) -> RGBAPixelArray | None:
            view: RCameraView = node.payload  # type: ignore[assignment]
            # Copies of a view show the picture of the same camera.
            return pictures.get(node.uid, pictures.get(("camera", view.camera_id)))

        target.reset(config)
        context = _CairoDrawingContext(
            camera=config.camera,
            target=target,
            image_resolver=resolve_image,
            reference_pixel_size=config.reference_pixel_size,
        )
        context.start_frame()
        self._draw_reusing(context, self._faces(nodes, config), config)
        return PixelFrame(target.read_pixels())

    def _target_for(self, config: RenderConfig) -> _CairoRenderTarget:
        self._state = RendererState.OPEN
        settings = _CairoRasterSettings(config.pixel_width, config.pixel_height)
        target = self._target
        if target is not None and target.settings != settings:
            target.close()
            target = self._target = None
        if target is None:
            target = self._target = _CairoRenderTarget(settings, self._image_files)
        return target

    def _draw_camera_views(self, config: RenderConfig) -> dict[Any, RGBAPixelArray]:
        """Draw each camera view into an image; return them by view uid and by
        ``("camera", camera_id)``.
        """
        pictures: dict[Any, RGBAPixelArray] = {}
        drawn = set()
        try:
            for view in config.camera_views:
                view_config = view.config
                settings = _CairoRasterSettings(
                    view_config.pixel_width, view_config.pixel_height
                )
                target = self._sub_targets.get(view.uid)
                if target is None or target.settings != settings:
                    if target is not None:
                        target.close()
                    target = self._sub_targets[view.uid] = _CairoRenderTarget(
                        settings, self._image_files
                    )
                target.reset(view_config)
                context = _CairoDrawingContext(
                    camera=view_config.camera,
                    target=target,
                    image_resolver=lambda node: None,
                    reference_pixel_size=view_config.reference_pixel_size,
                )
                context.start_frame()
                context.draw_nodes(self._faces(view.nodes, view_config))
                pictures[view.uid] = pictures[("camera", view.camera_id)] = (
                    target.pixels
                )
                drawn.add(view.uid)
        finally:
            # Retire targets of removed views and of views whose drawing failed.
            for key in self._sub_targets.keys() - drawn:
                self._sub_targets.pop(key).close()
        return pictures

    @staticmethod
    def _faces(nodes: Sequence[RNode], config: RenderConfig) -> list[RNode]:
        """``nodes`` with meshes as their faces, sorted far to near for a
        perspective camera: Cairo paints in order.
        """
        camera = config.camera
        drawn: list[RNode] = []
        for node in nodes:
            if isinstance(node.payload, RMesh):
                drawn.extend(mesh_faces(node, camera, config.reference_pixel_size))
            else:
                drawn.append(node)
        return sort_for_display(camera, drawn)

    def _draw_reusing(
        self,
        context: _CairoDrawingContext,
        nodes: list[RNode],
        config: RenderConfig,
    ) -> None:
        """Draw ``nodes`` over the target's background, copying the drawing of the
        leading nodes that did not change since they were saved.

        After a frame, the drawing of the nodes that did not change since the
        frame before is saved, to be copied in the next frames while they stay
        the same.
        """
        target = context.target
        keys = [node_key(node) for node in nodes]
        view = view_key(
            config.camera,
            (config.pixel_width, config.pixel_height),
            config.reference_pixel_size,
            _background_key(config),
        )
        memo = self._memo
        start = memo.reusable(keys, view)
        if start:
            assert self._saved_pixels is not None
            target.set_pixels(self._saved_pixels)
        else:
            save = reusable_boundary(nodes, memo.worth_saving(keys, view))
            if save:
                context.draw_nodes(nodes[:save])
                self._saved_pixels = target.read_pixels()
                memo.saved(keys[:save], view)
                start = save
            else:
                memo.forget_saved()
                self._saved_pixels = None
        context.draw_nodes(nodes[start:])
        memo.drew(keys, view)

    @contextlib.contextmanager
    def _scene_images(self) -> Iterator[None]:
        """See OpenGLRenderer._scene_images; Cairo images are cheap to set up."""
        yield

    def _image(self, nodes: Sequence[RNode], config: RenderConfig) -> RGBAPixelArray:
        """Draw a frame for inspection, apart from the frames of a render.

        A temporary renderer draws it, so that it works before the scene renders
        and after the renderer closed, and leaves the drawing kept for the next
        frame alone.
        """
        with CairoRenderer() as renderer:
            return renderer.render(nodes, config).pixels()

    def add_frame(self, frame: RGBAPixelArray, num_frames: int = 1) -> None:
        self._ensure_open()
        self._scene._get_manager()._legacy_add_frame(frame, num_frames)

    def freeze_current_frame(self, duration: float) -> None:
        self._ensure_open()
        manager = self._scene._get_manager()
        # An evaluation draws frames only when they are requested, without a target.
        manager._legacy_add_frame(
            self.render(*manager._frame()).pixels()
            if manager._mode.draws_added_frames
            else None,
            frame_count(duration, self._scene.session_spec.frame_rate),
        )

    def update_skipping_status(self) -> None:
        self._scene._get_manager()._update_skipping_status()

    def scene_finished(self, scene: Scene) -> None:
        self._ensure_open()
        output = self.file_writer.output_spec
        if self.num_plays and (output.is_video or output.is_image_sequence):
            self.file_writer.finish()
        if output.is_still or (not self.num_plays and output.fallback_to_still):
            # Skipped and cached plays do not draw: draw the final state.
            frame = self.render(*scene._get_manager()._frame())
            self.file_writer.save_image(frame.pixels())

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def open(self) -> None:
        """Check that the renderer can draw.

        Cairo creates its image buffers when it first draws, so there is nothing
        to allocate here. The method exists so that callers can open any
        renderer before :meth:`.Scene.setup` runs.
        """
        self._ensure_open()
        self._state = RendererState.OPEN

    def close(self) -> None:
        """Close the renderer and release its image buffers and drawing contexts.

        If releasing fails, for example on an interruption, the renderer is
        ``CLOSE_FAILED`` and ``close()`` can be called again.
        """
        if self._state is RendererState.CLOSED:
            return
        self._state = RendererState.CLOSING
        # Release everything before raising a failure or interruption. A target
        # that did not close stays, so that calling close() again closes it.
        failures: list[BaseException] = []

        def release(target: _CairoRenderTarget) -> bool:
            try:
                target.close()
            except BaseException as error:
                failures.append(error)
                return False
            return True

        if self._target is not None:
            release(self._target)
        for key, target in list(self._sub_targets.items()):
            if release(target):
                del self._sub_targets[key]
        self._image_files.clear()
        self._memo.clear()
        self._saved_pixels = None
        if failures:
            self._state = RendererState.CLOSE_FAILED
            for secondary in failures[1:]:
                logger.error("Additional Cairo cleanup failure", exc_info=secondary)
            raise failures[0]
        self._state = RendererState.CLOSED


def _background_key(config: RenderConfig) -> tuple[Any, ...]:
    return (
        config.background_image,
        tuple(config.background_color.to_rgba_with_alpha(config.background_opacity)),
    )
