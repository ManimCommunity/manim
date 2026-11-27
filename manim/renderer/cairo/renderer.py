from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self

from ..._config import logger
from .._reuse import PrefixMemo, node_key, view_key
from .._view import sort_for_display
from ..protocol import PixelFrame, RenderConfig, RendererState
from ..snapshot import RMesh
from ._mesh_faces import mesh_faces
from .rendering import _CairoDrawingContext
from .target import (
    _CairoRasterSettings,
    _CairoRenderTarget,
    _ImageFiles,
    background_key,
)

if TYPE_CHECKING:
    from ...typing import RGBAPixelArray
    from ..snapshot import RCameraView, RNode

__all__ = ["CairoRenderer"]


class CairoRenderer:
    """A renderer using Cairo.

    The renderer draws frames, including the pictures of camera views, into image
    buffers of the size each frame's :class:`.RenderConfig` asks for, allocated
    when it first draws one. It keeps the drawing of what did not change between
    frames.
    """

    def __init__(self) -> None:
        # Image files read for this renderer, shared by all its targets.
        self._image_files = _ImageFiles()
        self._target: _CairoRenderTarget | None = None
        self._sub_targets: dict[int, _CairoRenderTarget] = {}
        # The drawing of the leading nodes of a frame that did not change, and
        # their keys.
        self._memo = PrefixMemo()
        self._saved_pixels: RGBAPixelArray | None = None
        #: Cairo has no preview window.
        self.preview: None = None
        self._state = RendererState.UNOPENED

    @property
    def state(self) -> RendererState:
        """Where the renderer is in its life (read-only)."""
        return self._state

    def _ensure_open(self) -> None:
        """Check that the renderer is not closed."""
        if not self._state.accepts_work:
            raise RuntimeError(f"The Cairo renderer is {self._state.value}.")

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
            lights=config.lights,
            ambient=config.ambient,
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
                    lights=view_config.lights,
                    ambient=view_config.ambient,
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
                drawn.extend(
                    mesh_faces(
                        node,
                        camera,
                        config.reference_pixel_size,
                        config.lights,
                        config.ambient,
                    )
                )
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
            background_key(config),
            config.lights,
            config.ambient,
        )
        memo = self._memo
        start = memo.count_reusable(keys, view)
        if start:
            assert self._saved_pixels is not None
            target.set_pixels(self._saved_pixels)
        else:
            save = memo.count_worth_saving(keys, view)
            if save:
                context.draw_nodes(nodes[:save])
                self._saved_pixels = target.read_pixels()
                memo.mark_saved(keys[:save], view)
                start = save
            else:
                memo.forget_saved()
                self._saved_pixels = None
        context.draw_nodes(nodes[start:])
        memo.record_drawn(keys, view)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

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
