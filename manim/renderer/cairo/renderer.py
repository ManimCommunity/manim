from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from PIL import Image

from manim.utils.hashing import get_hash_from_play_call

from ... import config, logger
from ..._config.video_encoder import video_encoder_fingerprint
from ...camera.camera import Camera
from ...camera.multi_camera import MultiCamera
from ...mobject.mobject import Mobject, _AnimationBuilder
from ...mobject.types.image_mobject import ImageMobjectFromCamera
from ...scene.scene_file_writer import SceneFileWriter
from ...utils.exceptions import EndSceneEarlyException
from ...utils.iterables import list_update
from ..protocol import RendererCapabilities
from .rendering import _CairoDrawingContext
from .target import _CairoRasterSettings, _CairoRenderTarget

if TYPE_CHECKING:
    from manim._config.render_session import RenderSessionSpec
    from manim.animation.animation import Animation
    from manim.scene.scene import Scene
    from manim.scene.scene_file_writer import _SceneFileWriterSettings

    from ..typing import RGBAPixelArray

__all__ = ["CairoRenderer"]


class CairoRenderer:
    """A renderer using Cairo.

    Cameras supplied to this renderer contain semantic view/projection state only.
    CairoRenderer owns all pixel arrays, PyCairo contexts, nested targets, drawing,
    static raster reuse, and readback.
    """

    capabilities = RendererCapabilities(live_preview=False)

    def __init__(
        self,
        file_writer_class: type[SceneFileWriter] = SceneFileWriter,
        camera_class: type[Camera] | None = None,
        camera: Camera | None = None,
        skip_animations: bool = False,
        **kwargs: Any,
    ) -> None:
        if camera is not None and camera_class is not None:
            raise ValueError("Pass either camera or camera_class, not both.")
        self._file_writer_class = file_writer_class
        camera_cls = camera_class if camera_class is not None else Camera
        self.camera = camera if camera is not None else camera_cls()
        self._original_skipping_status = skip_animations
        self.skip_animations = skip_animations
        self.animations_hashes: list[str | None] = []
        self.num_plays = 0
        self.time = 0.0
        self._frame_rate = float(config["frame_rate"])
        settings = _CairoRasterSettings(
            pixel_width=int(config["pixel_width"]),
            pixel_height=int(config["pixel_height"]),
            base_pixel_width=int(config["pixel_width"]),
            base_pixel_height=int(config["pixel_height"]),
        )
        self._target = _CairoRenderTarget(settings)
        self._sub_targets: dict[int, _CairoRenderTarget] = {}
        self._camera_view_pixels: dict[int, RGBAPixelArray] = {}
        self.static_image: RGBAPixelArray | None = None
        self._render_all_mobjects = False

    def init_scene(
        self,
        scene: Scene,
        session_spec: RenderSessionSpec,
        file_writer_settings: _SceneFileWriterSettings,
    ) -> None:
        self.file_writer: Any = self._file_writer_class(file_writer_settings)

    def play(
        self,
        scene: Scene,
        *args: Animation | Mobject | _AnimationBuilder,
        **kwargs: Any,
    ) -> None:
        self.skip_animations = self._original_skipping_status
        self.update_skipping_status()
        scene.compile_animation_data(*args, **kwargs)

        if self.skip_animations:
            logger.debug(f"Skipping animation {self.num_plays}")
            hash_current_animation = None
            self.time += scene.duration
        else:
            if config["disable_caching"]:
                logger.info("Caching disabled.")
                hash_current_animation = f"uncached_{self.num_plays:05}"
            else:
                assert scene.animations is not None
                hash_current_animation = get_hash_from_play_call(
                    scene,
                    self.camera,
                    scene.animations,
                    scene.mobjects,
                    backend="cairo",
                    encoder_fingerprint=video_encoder_fingerprint(
                        scene.session_spec.video_encoder,
                    ),
                    renderer_state=(),
                )
                if self.file_writer.is_already_cached(hash_current_animation):
                    logger.info(
                        f"Animation {self.num_plays} : Using cached data (hash : %(hash_current_animation)s)",
                        {"hash_current_animation": hash_current_animation},
                    )
                    self.skip_animations = True
                    self.time += scene.duration
        self.file_writer.add_partial_movie_file(hash_current_animation)
        self.animations_hashes.append(hash_current_animation)
        logger.debug(
            "List of the first few animation hashes of the scene: %(h)s",
            {"h": str(self.animations_hashes[:5])},
        )

        self.file_writer.begin_animation(
            not self.skip_animations,
            animation_index=self.num_plays,
        )
        scene.begin_animations()
        self.save_static_frame_data(scene, scene.static_mobjects)

        if scene.is_current_animation_frozen_frame():
            self.update_frame(scene, mobjects=scene.moving_mobjects)
            self.freeze_current_frame(scene.duration)
        else:
            scene.play_internal()
        self.file_writer.end_animation(not self.skip_animations)
        self.num_plays += 1

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

    def update_frame(
        self,
        scene: Scene,
        mobjects: Iterable[Mobject] | None = None,
        include_submobjects: bool = True,
        ignore_skipping: bool = True,
        **kwargs: Any,
    ) -> None:
        """Render one scene state into the owned Cairo target."""
        if self.skip_animations and not ignore_skipping:
            return
        if not mobjects:
            mobjects = list_update(scene.mobjects, scene.foreground_mobjects)
        if self.static_image is not None:
            self._target.set_pixels(self.static_image)
        else:
            self._target.reset(self.camera)

        self._camera_view_pixels.clear()
        self._render_camera(
            camera=self.camera,
            target=self._target,
            mobjects=mobjects,
            include_submobjects=include_submobjects,
            excluded_mobjects=kwargs.get("excluded_mobjects"),
            camera_stack=(),
        )

    def render_mobjects(
        self,
        mobjects: Iterable[Mobject],
        *,
        camera: Camera | None = None,
    ) -> None:
        """Render explicit mobjects for direct image materialization."""
        render_camera = self.camera if camera is None else camera
        self._target.reset(render_camera)
        self._camera_view_pixels.clear()
        self._render_camera(
            camera=render_camera,
            target=self._target,
            mobjects=mobjects,
            include_submobjects=True,
            excluded_mobjects=None,
            camera_stack=(),
        )

    def render(
        self,
        scene: Scene,
        time: float,
        moving_mobjects: Iterable[Mobject] | None = None,
    ) -> None:
        if self._render_all_mobjects:
            moving_mobjects = None
        self.update_frame(scene, moving_mobjects)
        self.add_frame(self.get_frame())

    def get_frame(self) -> RGBAPixelArray:
        """Return a fresh owned top-left-origin RGBA frame."""
        return self._target.read_pixels()

    def get_image(self) -> Image.Image:
        """Return the current target as a PIL image."""
        return Image.fromarray(self.get_frame())

    def add_frame(self, frame: RGBAPixelArray, num_frames: int = 1) -> None:
        if self.skip_animations:
            return
        self.time += num_frames / self._frame_rate
        self.file_writer.write_frame(frame, repeat=num_frames)

    def freeze_current_frame(self, duration: float) -> None:
        self.add_frame(
            self.get_frame(),
            num_frames=int(duration * self._frame_rate),
        )

    def show_frame(self, scene: Scene) -> None:
        self.update_frame(scene, ignore_skipping=True)
        self.get_image().show()

    def save_static_frame_data(
        self,
        scene: Scene,
        static_mobjects: Iterable[Mobject],
    ) -> RGBAPixelArray | None:
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
        if self.file_writer.sections[-1].skip_animations:
            self.skip_animations = True
        if self.file_writer.output_spec.is_still:
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

    def scene_finished(self, scene: Scene) -> None:
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
        """Release all renderer-owned Cairo targets."""
        self._target.close()
        for target in self._sub_targets.values():
            target.close()
        self._sub_targets.clear()
        self._camera_view_pixels.clear()
