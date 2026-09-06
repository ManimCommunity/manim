"""Orchestration for rendering a scene."""

from __future__ import annotations

import datetime
import time
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import srt

from . import config, logger
from ._config.logger_utils import set_file_logger
from ._config.video_encoder import video_encoder_fingerprint
from .scene.section import DefaultSectionType
from .utils.exceptions import EndSceneEarlyException, RerunSceneException
from .utils.file_ops import open_media_file
from .utils.hashing import get_hash_from_play_call

if TYPE_CHECKING:
    from collections.abc import Callable
    from logging import FileHandler
    from types import TracebackType

    from PIL.Image import Image

    from ._config.output import OutputSpec
    from ._config.render_session import RenderSessionSpec
    from .animation.animation import Animation
    from .mobject.mobject import Mobject, _AnimationBuilder
    from .renderer.cairo import CairoRenderer
    from .renderer.cairo.camera import Camera
    from .renderer.opengl.camera import OpenGLCamera
    from .renderer.opengl.renderer import OpenGLRenderer
    from .scene.scene import Scene
    from .scene.scene_file_writer import SceneFileWriter
    from .typing import RGBAPixelArray

__all__ = ["Manager"]

SceneT = TypeVar("SceneT", bound="Scene")


class Manager(Generic[SceneT]):
    """Run a scene's rendering steps and clean up its resources.

    Each manager works with one :class:`~manim.scene.scene.Scene`. It calls the
    scene's setup, construct, and tear-down hooks, runs animation playback, and
    uses the renderer for drawing. A file writer handles sections, subcaptions,
    and audio. Calling :meth:`~manim.scene.scene.Scene.render` creates a manager
    if needed and stores it in ``scene.manager``.

    Parameters
    ----------
    scene
        The scene to render. Reuse ``scene.manager`` if a manager is already
        attached.

    Attributes
    ----------
    scene
        The managed scene.

    Notes
    -----
    Output settings are saved during scene construction. The manager creates
    the file writer when first requested and makes it available before
    :meth:`~manim.scene.scene.Scene.setup` runs.

    Both backends use the same animation clock and sampling rules. Cached and
    skipped animations follow a separate fast-forward path.

    Rendering normally closes the renderer before returning. Use a
    ``with manager:`` block to keep its resources available after a successful
    render, for example to read the last-rendered frame. The block closes them
    when it ends.

    Examples
    --------
    A scene normally creates its manager when :meth:`~manim.scene.scene.Scene.render`
    is called. It can also be attached explicitly::

        scene = Scene()
        manager = scene.manager or Manager(scene)
        manager.render()
    """

    def __init__(self, scene: SceneT) -> None:
        if scene.manager is not None:
            raise ValueError("A manager is already attached to this scene.")
        self.scene = scene
        self._execution = scene.renderer._claim_execution(self)
        self._file_writer: SceneFileWriter | None = None
        self._creating_file_writer = False
        self._log_handler: FileHandler | None = None
        self._closed = False
        self._closing = False
        self._scope_depth = 0
        scene.manager = self

    def __enter__(self) -> Manager[SceneT]:
        if self._closed or self._closing:
            raise RuntimeError("The Manager is closed or closing.")
        self._scope_depth += 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._scope_depth -= 1
        if exc is not None:
            self._cleanup_after_failure()
        elif self._scope_depth == 0:
            self.close()

    def _open_log_handler(self) -> None:
        if self._log_handler is not None:
            return
        path = self.scene._log_file_path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handler = set_file_logger(path)

    def _close_log_handler(self) -> None:
        handler = self._log_handler
        if handler is not None:
            logger.removeHandler(handler)
            handler.close()
            self._log_handler = None

    def close(self) -> None:
        """Stop encoding jobs, close the renderer, and close the scene's log file.

        This discards any unfinished segment and waits for queued encoding jobs.
        It removes the log handler created by this manager, leaving other handlers
        in place. Rendering calls this automatically, or at the end of an enclosing
        ``with manager:`` block after a successful render.
        """
        if self._closed:
            return
        self._closing = True
        # Attempt every cleanup even on KeyboardInterrupt/SystemExit, then reraise.
        failures: list[BaseException] = []

        def cleanup(callback: Callable[[], Any]) -> None:
            try:
                callback()
            except BaseException as error:
                failures.append(error)

        if self._file_writer is not None:
            cleanup(self._file_writer.abort_encode_jobs)

        cleanup(self.renderer.close)
        cleanup(self._close_log_handler)
        if failures:
            for secondary in failures[1:]:
                logger.error("Additional Manager cleanup failure", exc_info=secondary)
            raise failures[0]
        self._closed = True

    def _cleanup_after_failure(self) -> None:
        # A rendering exception is already propagating; cleanup must not replace it.
        try:
            self.close()
        except BaseException:
            logger.exception("Failed to clean up resources after a render failure")

    @property
    def renderer(self) -> CairoRenderer | OpenGLRenderer:
        """Return the renderer selected when the scene was created."""
        return self.scene.renderer

    @property
    def camera(self) -> Camera | OpenGLCamera:
        """Return the current renderer's camera."""
        return self.renderer.camera

    @property
    def file_writer(self) -> SceneFileWriter:
        """Return the scene's file writer, creating it on first access.

        The writer uses ``scene.file_writer_settings``. :meth:`render` accesses
        this property before calling :meth:`setup`; ``renderer.file_writer``
        returns the same writer. Both properties are read-only. To choose a custom
        writer class, pass ``file_writer_class`` when constructing the renderer.
        """
        if self._file_writer is None:
            if self._closed or self._closing:
                raise RuntimeError(
                    "Cannot create output resources on a closed Manager."
                )
            if self._creating_file_writer:
                raise RuntimeError("Recursive file writer creation is not supported.")
            self._creating_file_writer = True
            try:
                self._file_writer = self.renderer._file_writer_class(
                    self.scene.file_writer_settings
                )
            finally:
                self._creating_file_writer = False
        return self._file_writer

    @property
    def output_spec(self) -> OutputSpec:
        """Return the output format and options saved for this scene."""
        return self.session_spec.output

    @property
    def session_spec(self) -> RenderSessionSpec:
        """Return the scene's saved output, preview, and execution settings."""
        return self.scene.session_spec

    @property
    def time(self) -> float:
        """Return the managed execution time."""
        return self._execution.time

    @time.setter
    def time(self, value: float) -> None:
        self._execution.time = value

    @property
    def num_plays(self) -> int:
        """Return the managed execution's play count."""
        return self._execution.num_plays

    @num_plays.setter
    def num_plays(self, value: int) -> None:
        self._execution.num_plays = value

    @property
    def skip_animations(self) -> bool:
        """Return the managed execution's animation skip state."""
        return self._execution.skip_animations

    @skip_animations.setter
    def skip_animations(self, value: bool) -> None:
        self._execution.skip_animations = value

    def render(self, preview: bool = False) -> bool:
        """Run the complete render lifecycle for the managed scene.

        The lifecycle invokes :meth:`setup`, :meth:`construct`,
        :meth:`tear_down`, and :meth:`post_construct`, in that order. Reaching a
        configured animation boundary ends construction normally. After a
        successful render, the renderer closes before this method returns, or
        when an enclosing ``with manager:`` block ends.

        If setup, animation, output finalization, or opening the preview fails,
        the manager stops encoding jobs and closes the renderer before raising
        the original exception. This cleanup also applies to errors in
        :meth:`tear_down`.

        Parameters
        ----------
        preview
            Whether the rendered media should be opened after rendering.

        Returns
        -------
        bool
            ``True`` when an interactive rerun was requested; otherwise
            ``False``. This matches the return value of
            :meth:`~manim.scene.scene.Scene.render`.
        """
        from .renderer.opengl.renderer import OpenGLRenderer

        if self._closed or self._closing:
            raise RuntimeError("The Manager is closed or closing.")
        started = False
        try:
            presentation = self.session_spec.presentation
            open_after_render = preview or presentation.open_after_render
            if open_after_render and not self.output_spec.enabled:
                raise ValueError("Previewing after render requires a media artifact.")

            started = True
            self._open_log_handler()
            if isinstance(self.renderer, OpenGLRenderer):
                self.renderer.open()
            # Make the writer available to user setup() code.
            _ = self.file_writer
            self.setup()
            try:
                self.construct()
            except EndSceneEarlyException:
                # Only a construction boundary is a normal early scene end.
                pass
            except RerunSceneException:
                # The caller will create a fresh Scene for the rerun.
                # Report any encoder errors before allowing that restart.
                self.file_writer.abort_encode_jobs(reraise_encoder_failures=True)
                self._finish_resource_scope()
                return True
            self.tear_down()
            self.post_construct()

            if open_after_render or presentation.show_in_file_browser:
                open_media_file(
                    self.file_writer,
                    preview=open_after_render,
                    show_in_file_browser=presentation.show_in_file_browser,
                )

            self._finish_resource_scope()
            return False
        except BaseException:
            # An encoder waiting for more frames can prevent Python from exiting.
            # Stop it while preserving the original rendering error.
            if started or self._file_writer is not None:
                self._cleanup_after_failure()
            raise

    def _finish_resource_scope(self) -> None:
        self._close_log_handler()
        if self._scope_depth == 0:
            self.close()

    def get_image(self) -> Image:
        """Draw a fresh scene image; see :meth:`.Scene.get_image` for snapshot behavior."""
        return self.renderer._get_scene_image(self.scene)

    def setup(self) -> None:
        """Run the managed scene's :meth:`~manim.scene.scene.Scene.setup` hook."""
        self.scene.setup()

    def construct(self) -> None:
        """Run the managed scene's :meth:`~manim.scene.scene.Scene.construct` hook."""
        self.scene.construct()

    def post_construct(self) -> None:
        """Finalize output after scene construction and tear-down.

        This validates empty video output, asks the renderer to finish the scene,
        and logs the number of played animations. :meth:`tear_down` runs first,
        so its changes to the scene are included in last-frame image output.
        """
        output = self.output_spec
        empty_video_output = self.num_plays == 0 and output.is_video
        if empty_video_output and not output.fallback_to_still:
            raise RuntimeError(
                f"{self.scene} has no play calls, so the explicitly requested "
                f"{output.format.value.upper()} output cannot be produced. "
                "Use --format=png to save its last frame.",
            )

        self.renderer.scene_finished(self.scene)

        if (
            empty_video_output
            and output.fallback_to_still
            and getattr(self.file_writer, "final_file_path", None) is not None
        ):
            logger.warning(
                f"{self.scene} has no play calls. Automatic video output has "
                "been saved as a PNG instead.",
            )

        # Show info only if animations are rendered or to get image.
        if self.num_plays or output.enabled:
            logger.info(
                f"Rendered {str(self.scene)}\nPlayed {self.num_plays} animations",
            )

    def tear_down(self) -> None:
        """Run the managed scene's :meth:`~manim.scene.scene.Scene.tear_down` hook."""
        self.scene.tear_down()

    def play(
        self,
        *args: Animation | Mobject | _AnimationBuilder,
        subcaption: str | None = None,
        subcaption_duration: float | None = None,
        subcaption_offset: float = 0,
        **kwargs: Any,
    ) -> None:
        """Coordinate an animation request and its optional subcaption.

        Parameters
        ----------
        args
            Animations, mobjects, or animation builders passed by
            :meth:`~manim.scene.scene.Scene.play`.
        subcaption
            Content to add to the external subcaption file for this play call.
        subcaption_duration
            Duration of the subcaption. When omitted, the elapsed animation time
            is used.
        subcaption_offset
            Offset in seconds from the beginning of the play call.
        kwargs
            Additional animation arguments passed to Scene's compilation helpers.
        """
        if self._closed or self._closing:
            raise RuntimeError("The Manager is closed or closing.")
        try:
            self._open_log_handler()
            start_time = self.time
            self._play(*args, **kwargs)
            run_time = self.time - start_time

            if subcaption:
                if subcaption_duration is None:
                    subcaption_duration = run_time
                # Place the caption relative to the start of this play, through
                # Scene's public API so its customization hook is preserved.
                self.scene.add_subcaption(
                    content=subcaption,
                    duration=subcaption_duration,
                    offset=-run_time + subcaption_offset,
                )
        except (EndSceneEarlyException, RerunSceneException):
            # Leave early scene endings and rerun requests for the caller to handle.
            raise
        except BaseException:
            # A direct play() can fail outside render(), so it needs cleanup here too.
            self._cleanup_after_failure()
            raise

    def _play(
        self, *args: Animation | Mobject | _AnimationBuilder, **kwargs: Any
    ) -> None:
        from .renderer.opengl.renderer import OpenGLRenderer

        if isinstance(self.renderer, OpenGLRenderer):
            self._play_opengl(*args, **kwargs)
        else:
            self._play_cairo(*args, **kwargs)

    def _update_skipping_status(self) -> None:
        if (
            self.file_writer.sections[-1].skip_animations
            or self.file_writer.output_spec.is_still
        ):
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

    def _play_cairo(
        self, *args: Animation | Mobject | _AnimationBuilder, **kwargs: Any
    ) -> None:
        """Preserve Cairo's existing order while moving its orchestration owner."""
        scene = self.scene
        renderer = cast("CairoRenderer", self.renderer)
        renderer._ensure_open()
        self.skip_animations = renderer._original_skipping_status
        self._update_skipping_status()
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
                        self.session_spec.video_encoder
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
        renderer.animations_hashes.append(hash_current_animation)
        logger.debug(
            "List of the first few animation hashes of the scene: %(h)s",
            {"h": str(renderer.animations_hashes[:5])},
        )
        self.file_writer.begin_animation(
            not self.skip_animations, animation_index=self.num_plays
        )
        scene.begin_animations()
        renderer.save_static_frame_data(scene, scene.static_mobjects)
        if scene.is_current_animation_frozen_frame():
            renderer.update_frame(scene, mobjects=scene.moving_mobjects)
            frame = renderer.get_frame()
            frame_rate = float(config.frame_rate)
            repeats = int(scene.duration * frame_rate)
            if not self.skip_animations:
                self.time += repeats / frame_rate
                self.file_writer.write_frame(frame, repeat=repeats)
        else:
            scene.play_internal()
        self.file_writer.end_animation(not self.skip_animations)
        self.num_plays += 1

    def _play_opengl(
        self, *args: Animation | Mobject | _AnimationBuilder, **kwargs: Any
    ) -> None:
        """Keep the former decorator/body ordering, including double compilation."""
        scene = self.scene
        renderer = cast("OpenGLRenderer", self.renderer)
        self.skip_animations = renderer._original_skipping_status
        self._update_skipping_status()
        animations = scene.compile_animations(*args, **kwargs)
        scene.add_mobjects_from_animations(animations)
        skipped_at_entry = self.skip_animations
        if skipped_at_entry:
            logger.debug(f"Skipping animation {self.num_plays}")
        else:
            if not config["disable_caching"]:
                hash_play = get_hash_from_play_call(
                    scene,
                    self.camera,
                    animations,
                    scene.mobjects,
                    backend="opengl",
                    encoder_fingerprint=video_encoder_fingerprint(
                        self.session_spec.video_encoder
                    ),
                    renderer_state={
                        "meshes": scene.meshes,
                        "background_color": renderer.background_color,
                        "anti_alias_width": renderer.anti_alias_width,
                        # Time-dependent user code now sees sample time instead
                        # of the old event-start time: old pixels are not reusable.
                        "execution_clock": "sample-v1",
                    },
                )
                if self.file_writer.is_already_cached(hash_play):
                    logger.info(
                        f"Animation {self.num_plays} : Using cached data (hash : %(hash_play)s)",
                        {"hash_play": hash_play},
                    )
                    self.skip_animations = True
            else:
                hash_play = f"uncached_{self.num_plays:05}"
            renderer.animations_hashes.append(hash_play)
            self.file_writer.add_partial_movie_file(hash_play)
            logger.debug(
                "List of the first few animation hashes of the scene: %(h)s",
                {"h": str(renderer.animations_hashes[:5])},
            )

        renderer.open()
        renderer.animation_start_time = time.time()
        self.file_writer.begin_animation(
            not self.skip_animations, animation_index=self.num_plays
        )
        scene.compile_animation_data(*args, **kwargs)
        if self.skip_animations:
            self.time += scene.duration
        scene.begin_animations()
        if scene.is_current_animation_frozen_frame():
            renderer.update_frame(scene)
            output = self.file_writer.output_spec
            frame_rate = float(config.frame_rate)
            repeats = int(scene.duration * frame_rate)
            frame = (
                renderer.get_frame()
                if not self.skip_animations
                and (output.is_video or output.is_image_sequence)
                else None
            )
            if not self.skip_animations:
                self.time += repeats / frame_rate
                if frame is not None:
                    self.file_writer.write_frame(frame, repeat=repeats)
            if renderer.window is not None:
                renderer.window.swap_buffers()
                while time.time() - renderer.animation_start_time < scene.duration:
                    pass
            renderer.animation_elapsed_time = scene.duration
        else:
            scene.play_internal()
        self.file_writer.end_animation(not self.skip_animations)
        self.num_plays += 1
        if skipped_at_entry:
            renderer.animations_hashes.append(None)
            self.file_writer.add_partial_movie_file(None)

    def _play_internal(self, skip_rendering: bool = False) -> None:
        """Evaluate samples on the common clock, independently of pixel delivery."""
        scene = self.scene
        # Use the same rate as Scene.get_time_progression, not a backend's
        # raster-target settings. Resolution timing and sample rounding stay put.
        sample_step = 1 / config.frame_rate
        assert scene.animations is not None
        scene.duration = scene.get_run_time(scene.animations)
        scene.time_progression = scene._get_animation_time_progression(
            scene.animations,
            scene.duration,
        )
        for t in scene.time_progression:
            scene.update_to_time(t)
            draw = not skip_rendering and not scene.skip_animation_preview
            frame = self._draw_animation_frame(t) if draw else None
            # The sample represents one interval, including the initial t=0
            # sample. Stop conditions and finish observe the consumed span.
            if not self.skip_animations:
                self.time += sample_step
            if draw:
                self._deliver_animation_frame(frame, t)
            if scene.stop_condition is not None and scene.stop_condition():
                scene.time_progression.close()
                break
        for animation in scene.animations:
            animation.finish()
            animation.clean_up_from_scene(scene)
        if not self.skip_animations:
            scene.update_mobjects(0)
        self.renderer.static_image = None  # type: ignore[union-attr]
        scene.time_progression.close()

    def _draw_animation_frame(self, frame_offset: float) -> RGBAPixelArray | None:
        from .renderer.cairo.renderer import CairoRenderer

        self.renderer.render(self.scene, frame_offset, self.scene.moving_mobjects)
        if isinstance(self.renderer, CairoRenderer) or (
            not self.skip_animations
            and (
                self.file_writer.output_spec.is_video
                or self.file_writer.output_spec.is_image_sequence
            )
        ):
            return self.renderer.get_frame()
        return None

    def _deliver_animation_frame(
        self, frame: RGBAPixelArray | None, frame_offset: float
    ) -> None:
        from .renderer.opengl.renderer import OpenGLRenderer

        if self.skip_animations:
            return
        if frame is not None:
            self.file_writer.write_frame(frame)
        if isinstance(self.renderer, OpenGLRenderer):
            self.renderer._present_frame(self.scene, frame_offset)

    def _legacy_add_frame(self, frame: RGBAPixelArray, num_frames: int = 1) -> None:
        """Compatibility for explicit Cairo add_frame calls, not evaluation."""
        renderer = cast("CairoRenderer", self.renderer)
        renderer._ensure_open()
        if not self.skip_animations:
            self.time += num_frames / renderer._frame_rate
            self.file_writer.write_frame(frame, repeat=num_frames)

    def next_section(
        self,
        name: str = "unnamed",
        section_type: str = DefaultSectionType.NORMAL,
        skip_animations: bool = False,
    ) -> None:
        """Create a new output section.

        Parameters
        ----------
        name
            The section name.
        section_type
            The section type stored in the section manifest.
        skip_animations
            Whether animation output in this section should be skipped.
        """
        self.file_writer.next_section(name, section_type, skip_animations)

    def add_subcaption(
        self, content: str, duration: float = 1, offset: float = 0
    ) -> None:
        """Add a subcaption at the current scene time.

        Parameters
        ----------
        content
            The subcaption text.
        duration
            The duration in seconds for which the subcaption is displayed.
        offset
            The offset in seconds from the current scene time.
        """
        subtitle = srt.Subtitle(
            index=len(self.file_writer.subcaptions),
            content=content,
            start=datetime.timedelta(seconds=float(self.time + offset)),
            end=datetime.timedelta(seconds=float(self.time + offset + duration)),
        )
        self.file_writer.subcaptions.append(subtitle)

    def add_sound(
        self,
        sound_file: str,
        time_offset: float = 0,
        gain: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Add sound to the output at the current scene time.

        No sound is added while animations are being skipped.

        Parameters
        ----------
        sound_file
            The path to the sound file.
        time_offset
            The offset in seconds from the current scene time.
        gain
            The gain adjustment applied to the sound.
        kwargs
            Additional arguments forwarded to
            :meth:`~manim.scene.scene_file_writer.SceneFileWriter.add_sound`.
        """
        if self.skip_animations:
            return
        self.file_writer.add_sound(sound_file, self.time + time_offset, gain, **kwargs)
