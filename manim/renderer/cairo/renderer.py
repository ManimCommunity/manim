from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any, Self

from PIL import Image

from manim.camera import Camera, MultiCamera

from ..._config import config, logger
from ..._run.frames import frame_count
from ...mobject.mobject import Mobject, _AnimationBuilder
from ...mobject.types.image_mobject import ImageMobjectFromCamera
from ...scene.scene_file_writer import SceneFileWriter
from ...utils.iterables import list_update
from .._execution import _RendererExecutionView
from .._reuse import PrefixMemo, node_key, view_key
from ..extract import Extractor
from ..protocol import RenderConfig, RendererCapabilities, RendererState
from .rendering import _CairoDrawingContext, reusable_boundary
from .target import _CairoRasterSettings, _CairoRenderTarget, _ImageFiles

if TYPE_CHECKING:
    from manim._config.render_session import RenderSessionSpec
    from manim.animation.animation import Animation
    from manim.scene.scene import Scene

    from ...typing import RGBAPixelArray
    from ..snapshot import RCameraView, RNode

__all__ = ["CairoRenderer"]


class CairoRenderer(_RendererExecutionView):
    """A renderer using Cairo.

    The renderer draws the camera's view, including inset views, into image
    buffers. It manages the PyCairo drawing contexts and caches images of
    stationary mobjects. Construction saves the pixel dimensions; the buffers
    are allocated on the first draw or request for pixels.
    """

    capabilities = RendererCapabilities(live_preview=False)

    def __init__(
        self,
        camera_class: type[Camera] | None = None,
        camera: Camera | None = None,
        skip_animations: bool = False,
        *,
        _raster_settings: _CairoRasterSettings | None = None,
    ) -> None:
        if camera is not None and camera_class is not None:
            raise ValueError("Pass either camera or camera_class, not both.")
        if camera is None:
            camera = Camera() if camera_class is None else camera_class()
        self.camera = camera
        self._initialize_execution(skip_animations)
        self._raster_settings = _raster_settings or _CairoRasterSettings(
            pixel_width=config.pixel_width,
            pixel_height=config.pixel_height,
            base_pixel_width=config.pixel_width,
            base_pixel_height=config.pixel_height,
        )
        # Image files read for this renderer, shared by all its targets.
        self._image_files = _ImageFiles()
        self._target: _CairoRenderTarget | None = None
        # A finished scene whose final state is drawn on the first pixel read.
        self._final_scene: Scene | None = None
        self._sub_targets: dict[int, _CairoRenderTarget] = {}
        self._camera_view_pixels: dict[int, RGBAPixelArray] = {}
        # The drawing of the leading nodes of a frame of the whole scene that did
        # not change, and their keys.
        self._memo = PrefixMemo()
        self._extractor = Extractor()
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

    def _get_target(self) -> _CairoRenderTarget:
        self._ensure_open()
        self._state = RendererState.OPEN
        if self._target is None:
            self._target = _CairoRenderTarget(self._raster_settings, self._image_files)
        return self._target

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

    def _sub_target_for(
        self,
        image_mobject: ImageMobjectFromCamera,
        *,
        parent_camera: Camera,
        parent_target: _CairoRenderTarget,
    ) -> _CairoRenderTarget:
        parent_settings = parent_target.settings
        pixel_height = max(
            1,
            int(
                parent_settings.pixel_height
                * image_mobject.height
                / parent_camera.frame_height
            ),
        )
        pixel_width = max(
            1,
            int(
                parent_settings.pixel_width
                * image_mobject.width
                / parent_camera.frame_width
            ),
        )
        key = image_mobject.uid
        target = self._sub_targets.get(key)
        if target is not None and (
            target.settings.pixel_width != pixel_width
            or target.settings.pixel_height != pixel_height
        ):
            target.close()
            target = None
        if target is None:
            target = _CairoRenderTarget(
                parent_settings.resized(
                    pixel_width=pixel_width,
                    pixel_height=pixel_height,
                ),
                self._image_files,
            )
            self._sub_targets[key] = target
        return target

    def _draw_frame(
        self,
        *,
        camera: Camera,
        mobjects: Iterable[Mobject],
        include_submobjects: bool = True,
        excluded_mobjects: list[Mobject] | None = None,
        reuse: bool = False,
        background: Any = None,
    ) -> None:
        """Draw what ``camera`` sees, camera views first; views show no views."""
        target = self._get_target()
        mobject_list = list(mobjects)
        self._camera_view_pixels.clear()
        camera_view_pixels: dict[int, RGBAPixelArray] = {}
        try:
            if isinstance(camera, MultiCamera):
                for image_mobject in camera.image_mobjects_from_cameras:
                    sub_target = self._sub_target_for(
                        image_mobject, parent_camera=camera, parent_target=target
                    )
                    view_camera = image_mobject.camera
                    sub_target.reset(_render_config(view_camera, sub_target))
                    _CairoDrawingContext(
                        camera=view_camera._state(),
                        target=sub_target,
                        image_resolver=lambda node: None,
                        use_z_index=view_camera.use_z_index,
                        extractor=self._extractor,
                    ).draw(
                        mobject_list,
                        include_submobjects=include_submobjects,
                        excluded_mobjects=list_update(
                            list(excluded_mobjects or []), [image_mobject]
                        ),
                    )
                    self._camera_view_pixels[image_mobject.uid] = sub_target.pixels
                    camera_view_pixels[id(view_camera)] = sub_target.pixels
        finally:
            # Only completed views lend pixels to this frame. Retire targets for
            # removed views and any view whose drawing failed before completion.
            unused = self._sub_targets.keys() - self._camera_view_pixels.keys()
            for key in unused:
                self._sub_targets.pop(key).close()

        def resolve_image(node: RNode) -> RGBAPixelArray | None:
            # Unregistered views, such as copies of a registered view, show the
            # pixels most recently drawn for their camera.
            view: RCameraView = node.payload  # type: ignore[assignment]
            pixels = self._camera_view_pixels.get(node.uid)
            if pixels is None:
                pixels = camera_view_pixels.get(view.camera_id)
            return pixels

        context = _CairoDrawingContext(
            camera=camera._state(),
            target=target,
            image_resolver=resolve_image,
            use_z_index=camera.use_z_index,
            extractor=self._extractor,
        )
        context.start_frame()
        nodes = context.nodes(
            mobject_list,
            include_submobjects=include_submobjects,
            excluded_mobjects=excluded_mobjects,
        )
        if not reuse:
            context.draw_nodes(nodes)
            return
        self._draw_reusing(context, nodes, background)

    def _draw_reusing(
        self,
        context: _CairoDrawingContext,
        nodes: list[RNode],
        background: Any,
    ) -> None:
        """Draw ``nodes`` over the target's background, copying the drawing of the
        leading nodes that did not change since they were saved.

        After a frame, the drawing of the nodes that did not change since the
        frame before is saved, to be copied in the next frames while they stay
        the same.
        """
        target = context.target
        settings = target.settings
        keys = [node_key(node) for node in nodes]
        view = view_key(
            context.camera, (settings.pixel_width, settings.pixel_height), background
        )
        memo = self._memo
        start = memo.reusable(keys, view)
        if start:
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

    def update_frame(
        self,
        scene: Scene,
        mobjects: Iterable[Mobject] | None = None,
        include_submobjects: bool = True,
        ignore_skipping: bool = True,
        *,
        excluded_mobjects: list[Mobject] | None = None,
    ) -> None:
        """Draw the scene's current state into the renderer's image buffer.

        ``excluded_mobjects`` and their families are left out of the frame.
        """
        self._ensure_open()
        if self.skip_animations and not ignore_skipping:
            return
        self._final_scene = None
        mobjects_given = mobjects or None
        if not mobjects:
            mobjects = list_update(scene.mobjects, scene.foreground_mobjects)
        target = self._get_target()
        config = _render_config(self.camera, target)
        target.reset(config)
        self._draw_frame(
            camera=self.camera,
            mobjects=mobjects,
            include_submobjects=include_submobjects,
            excluded_mobjects=excluded_mobjects,
            # Reuse only between frames of the whole scene: a frame of some
            # mobjects only has other nodes.
            reuse=mobjects_given is None,
            background=_background_key(config),
        )

    def render_mobjects(
        self,
        mobjects: Iterable[Mobject],
        *,
        camera: Camera | None = None,
    ) -> None:
        """Draw the supplied mobjects into the renderer's image buffer."""
        self._ensure_open()
        render_camera = self.camera if camera is None else camera
        self._final_scene = None
        target = self._get_target()
        target.reset(_render_config(render_camera, target))
        self._draw_frame(camera=render_camera, mobjects=mobjects)

    def render(
        self,
        scene: Scene,
        time: float,
    ) -> None:
        # Manager advances animation time and sends the pixels to the writer.
        self.update_frame(scene)

    def get_frame(self) -> RGBAPixelArray:
        """Copy the current image to an RGBA array, with row zero at the top."""
        self._draw_final_state()
        return self._get_target().read_pixels()

    def _draw_final_state(self) -> None:
        scene, self._final_scene = self._final_scene, None
        if scene is not None:
            self.update_frame(scene)

    @contextlib.contextmanager
    def _scene_images(self) -> Iterator[None]:
        """See OpenGLRenderer._scene_images; Cairo images are cheap to set up."""
        yield

    def _get_scene_image(self, scene: Scene) -> Image.Image:
        """Draw a snapshot with a temporary renderer at the existing resolution."""
        with CairoRenderer(
            camera=self.camera, _raster_settings=self._raster_settings
        ) as renderer:
            renderer.render_mobjects(
                list_update(scene.mobjects, scene.foreground_mobjects)
            )
            return renderer.get_image()

    def get_image(self) -> Image.Image:
        """Return the current target as a PIL image."""
        return Image.fromarray(self.get_frame())

    def add_frame(self, frame: RGBAPixelArray, num_frames: int = 1) -> None:
        self._ensure_open()
        self._scene._get_manager()._legacy_add_frame(frame, num_frames)

    def freeze_current_frame(self, duration: float) -> None:
        self._ensure_open()
        manager = self._scene._get_manager()
        # An evaluation draws frames only when they are requested, without a target.
        manager._legacy_add_frame(
            self.get_frame() if manager._mode.draws_added_frames else None,
            frame_count(duration, self._scene.session_spec.frame_rate),
        )

    def show_frame(self, scene: Scene) -> None:
        self.update_frame(scene, ignore_skipping=True)
        self.get_image().show()

    def update_skipping_status(self) -> None:
        self._scene._get_manager()._update_skipping_status()

    def scene_finished(self, scene: Scene) -> None:
        self._ensure_open()
        output = self.file_writer.output_spec
        if self.num_plays and (output.is_video or output.is_image_sequence):
            self.file_writer.finish()
        # Skipped and cached plays do not draw, so the buffer may not show the
        # final state. Draw it when the pixels are read, not for every scene.
        self._final_scene = scene

        if output.is_still or (not self.num_plays and output.fallback_to_still):
            self.file_writer.save_image(self.get_frame())

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
        self._camera_view_pixels.clear()
        self._image_files.clear()
        self._memo.clear()
        self._saved_pixels = None
        self._final_scene = None
        if failures:
            self._state = RendererState.CLOSE_FAILED
            for secondary in failures[1:]:
                logger.error("Additional Cairo cleanup failure", exc_info=secondary)
            raise failures[0]
        self._state = RendererState.CLOSED


def _render_config(camera: Camera, target: _CairoRenderTarget) -> RenderConfig:
    settings = target.settings
    return RenderConfig(
        pixel_width=settings.pixel_width,
        pixel_height=settings.pixel_height,
        background_color=camera.background_color,
        background_opacity=camera.background_opacity,
        camera=camera._state(),
        background_image=camera.background_image,
    )


def _background_key(config: RenderConfig) -> tuple[Any, ...]:
    return (
        config.background_image,
        tuple(config.background_color.to_rgba_with_alpha(config.background_opacity)),
    )
