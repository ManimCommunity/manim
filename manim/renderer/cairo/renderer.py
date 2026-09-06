from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from PIL import Image

from ... import config
from ...mobject.mobject import Mobject, _AnimationBuilder
from ...mobject.types.image_mobject import ImageMobjectFromCamera
from ...scene.scene_file_writer import SceneFileWriter
from ...utils.iterables import list_update
from .._execution import _RendererExecutionView
from ..protocol import RendererCapabilities
from .camera import Camera, MultiCamera
from .rendering import _CairoDrawingContext
from .target import _CairoRasterSettings, _CairoRenderTarget

if TYPE_CHECKING:
    from manim._config.render_session import RenderSessionSpec
    from manim.animation.animation import Animation
    from manim.scene.scene import Scene

    from ...typing import RGBAPixelArray

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
        file_writer_class: type[SceneFileWriter] = SceneFileWriter,
        camera_class: type[Camera] | None = None,
        camera: Camera | None = None,
        skip_animations: bool = False,
        *,
        _raster_settings: _CairoRasterSettings | None = None,
    ) -> None:
        if camera is not None and camera_class is not None:
            raise ValueError("Pass either camera or camera_class, not both.")
        self._file_writer_class = file_writer_class
        camera_cls = camera_class if camera_class is not None else Camera
        self.camera = camera if camera is not None else camera_cls()
        self._initialize_execution(skip_animations)
        self._raster_settings = _raster_settings or _CairoRasterSettings(
            pixel_width=int(config["pixel_width"]),
            pixel_height=int(config["pixel_height"]),
            base_pixel_width=int(config["pixel_width"]),
            base_pixel_height=int(config["pixel_height"]),
        )
        self._target: _CairoRenderTarget | None = None
        self._sub_targets: dict[int, _CairoRenderTarget] = {}
        self._camera_view_pixels: dict[int, RGBAPixelArray] = {}
        self.static_image: RGBAPixelArray | None = None
        self._render_all_mobjects = False
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("The Cairo renderer is closed.")

    def _get_target(self) -> _CairoRenderTarget:
        self._ensure_open()
        if self._target is None:
            self._target = _CairoRenderTarget(self._raster_settings)
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
        """Compatibility entrypoint; Manager owns animation orchestration."""
        scene._get_manager()._play(*args, **kwargs)

    def _animation_cache_identity(self, scene: Scene) -> tuple[str, Any]:
        return "cairo", ()

    def _start_animation(self) -> None:
        self._ensure_open()

    def _prepare_animation(self, scene: Scene) -> None:
        self.save_static_frame_data(scene, scene.static_mobjects)

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
        key = id(image_mobject)
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
            )
            self._sub_targets[key] = target
        return target

    def _render_camera(
        self,
        *,
        camera: Camera,
        target: _CairoRenderTarget,
        mobjects: Iterable[Mobject],
        include_submobjects: bool,
        excluded_mobjects: list[Mobject] | None,
        camera_stack: tuple[int, ...],
    ) -> None:
        camera_id = id(camera)
        if camera_id in camera_stack:
            raise RuntimeError("Cairo camera views cannot contain a composition cycle.")
        next_stack = (*camera_stack, camera_id)
        mobject_list = list(mobjects)

        if isinstance(camera, MultiCamera):
            for image_mobject in camera.image_mobjects_from_cameras:
                sub_target = self._sub_target_for(
                    image_mobject,
                    parent_camera=camera,
                    parent_target=target,
                )
                sub_target.reset(image_mobject.camera)
                sub_excluded_mobjects = list_update(
                    list(excluded_mobjects or []),
                    [image_mobject],
                )
                self._render_camera(
                    camera=image_mobject.camera,
                    target=sub_target,
                    mobjects=mobject_list,
                    include_submobjects=include_submobjects,
                    excluded_mobjects=sub_excluded_mobjects,
                    camera_stack=next_stack,
                )
                self._camera_view_pixels[id(image_mobject)] = sub_target.pixels

        def resolve_image(
            image_mobject: ImageMobjectFromCamera,
        ) -> RGBAPixelArray | None:
            return self._camera_view_pixels.get(id(image_mobject))

        _CairoDrawingContext(
            camera=camera,
            target=target,
            image_resolver=resolve_image,
        ).draw(
            mobject_list,
            include_submobjects=include_submobjects,
            excluded_mobjects=excluded_mobjects,
        )

    def _draw_frame(
        self,
        *,
        camera: Camera,
        mobjects: Iterable[Mobject],
        include_submobjects: bool = True,
        excluded_mobjects: list[Mobject] | None = None,
    ) -> None:
        self._camera_view_pixels.clear()
        try:
            self._render_camera(
                camera=camera,
                target=self._get_target(),
                mobjects=mobjects,
                include_submobjects=include_submobjects,
                excluded_mobjects=excluded_mobjects,
                camera_stack=(),
            )
        finally:
            # Only completed views lend pixels to this frame. Retire targets for
            # removed views and any view whose drawing failed before completion.
            unused = self._sub_targets.keys() - self._camera_view_pixels.keys()
            for key in unused:
                self._sub_targets.pop(key).close()

    def update_frame(
        self,
        scene: Scene,
        mobjects: Iterable[Mobject] | None = None,
        include_submobjects: bool = True,
        ignore_skipping: bool = True,
        **kwargs: Any,
    ) -> None:
        """Draw the scene's current state into the renderer's image buffer."""
        self._ensure_open()
        if self.skip_animations and not ignore_skipping:
            return
        if not mobjects:
            mobjects = list_update(scene.mobjects, scene.foreground_mobjects)
        target = self._get_target()
        if self.static_image is not None:
            target.set_pixels(self.static_image)
        else:
            target.reset(self.camera)

        self._draw_frame(
            camera=self.camera,
            mobjects=mobjects,
            include_submobjects=include_submobjects,
            excluded_mobjects=kwargs.get("excluded_mobjects"),
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
        self._get_target().reset(render_camera)
        self._draw_frame(camera=render_camera, mobjects=mobjects)

    def render(
        self,
        scene: Scene,
        time: float,
        moving_mobjects: Iterable[Mobject] | None = None,
    ) -> None:
        if self._render_all_mobjects:
            moving_mobjects = None
        # Drawing is separate from Manager-owned time advancement and delivery.
        self.update_frame(scene, moving_mobjects)

    def get_frame(self) -> RGBAPixelArray:
        """Copy the current image to an RGBA array, with row zero at the top."""
        return self._get_target().read_pixels()

    def _get_scene_image(self, scene: Scene) -> Image.Image:
        """Draw a snapshot with a temporary renderer at the existing resolution."""
        renderer = CairoRenderer(
            camera=self.camera, _raster_settings=self._raster_settings
        )
        try:
            renderer.render_mobjects(
                list_update(scene.mobjects, scene.foreground_mobjects)
            )
            return renderer.get_image()
        finally:
            renderer.close()

    def get_image(self) -> Image.Image:
        """Return the current target as a PIL image."""
        return Image.fromarray(self.get_frame())

    def add_frame(self, frame: RGBAPixelArray, num_frames: int = 1) -> None:
        self._ensure_open()
        self._scene._get_manager()._legacy_add_frame(frame, num_frames)

    def freeze_current_frame(self, duration: float) -> None:
        self.add_frame(
            self.get_frame(),
            num_frames=int(duration * self._scene.session_spec.frame_rate),
        )

    def show_frame(self, scene: Scene) -> None:
        self.update_frame(scene, ignore_skipping=True)
        self.get_image().show()

    def save_static_frame_data(
        self,
        scene: Scene,
        static_mobjects: Iterable[Mobject],
    ) -> RGBAPixelArray | None:
        self._ensure_open()
        self.static_image = None
        # A nested view can contain any dynamic scene mobject regardless of the
        # view's position in the primary draw order. Keep all primary inputs
        # dynamic until the Manager cutover supplies explicit dynamic roots.
        self._render_all_mobjects = isinstance(self.camera, MultiCamera) and bool(
            scene.moving_mobjects,
        )
        if self._render_all_mobjects or not static_mobjects:
            return None
        self.update_frame(scene, mobjects=static_mobjects)
        self.static_image = self.get_frame()
        return self.static_image

    def update_skipping_status(self) -> None:
        self._scene._get_manager()._update_skipping_status()

    def scene_finished(self, scene: Scene) -> None:
        self._ensure_open()
        output = self.file_writer.output_spec
        if self.num_plays and (output.is_video or output.is_image_sequence):
            self.file_writer.finish()
        elif not self.num_plays:
            self.static_image = None
            self.update_frame(scene)

        if output.is_still or (not self.num_plays and output.fallback_to_still):
            if self.num_plays:
                self.static_image = None
                self.update_frame(scene)
            self.file_writer.save_image(self.get_frame())

    def close(self) -> None:
        """Close the renderer and release its image buffers and drawing contexts."""
        if self._closed:
            return
        self._closed = True
        if self._target is not None:
            self._target.close()
        for target in self._sub_targets.values():
            target.close()
        self._sub_targets.clear()
        self._camera_view_pixels.clear()
        self.static_image = None
        self._render_all_mobjects = False
