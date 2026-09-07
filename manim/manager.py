"""Orchestration for rendering a scene."""

from __future__ import annotations

import contextlib
import datetime
import math
from typing import TYPE_CHECKING, Any, Generic, TypeVar

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
    from .renderer.protocol import _AnimationRenderer
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
        self._evaluating = False
        self._evaluation_started = False
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
        if self._evaluating:
            return
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
        if self._evaluating:
            raise RuntimeError(
                "Media output is unavailable during no-raster evaluation."
            )
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
        """Return elapsed animation time in seconds."""
        return self._execution.time

    @time.setter
    def time(self, value: float) -> None:
        self._execution.time = value

    @property
    def num_plays(self) -> int:
        """Return the number of completed play or wait calls."""
        return self._execution.num_plays

    @num_plays.setter
    def num_plays(self, value: int) -> None:
        self._execution.num_plays = value

    @property
    def skip_animations(self) -> bool:
        """Return whether the current play is being fast-forwarded."""
        return self._execution.skip_animations

    @skip_animations.setter
    def skip_animations(self, value: bool) -> None:
        self._execution.skip_animations = value

    def _validate_execution(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("The Manager is closed or closing.")
        if self.renderer._closed or getattr(self.renderer, "_retiring", False):
            raise RuntimeError("Cannot execute a closed or retiring renderer.")
        if float(config.frame_rate) != self.session_spec.frame_rate:
            raise ValueError(
                "frame_rate changed after Scene construction. Construct a new Scene "
                "under the desired configuration before executing it."
            )

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

        if self._evaluating:
            raise RuntimeError("Cannot render during no-raster evaluation.")
        self._validate_execution()
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

    def evaluate(self) -> None:
        """Run setup, construction and teardown without engine pixels or media.

        Uses normal interpolation/updaters/stop conditions and the same clock as
        uncached rendering. Cache, skip and range selection are ignored. Explicit
        bound-scene image/GPU/output requests are rejected. Arbitrary user code is
        still executed: this is not a sandbox or a guarantee of zero user I/O.

        Start with an unused, cold Scene. Time and scene state remain inspectable;
        no file writer, encoder, native host or execution log is implicitly opened.
        """
        self._validate_execution()
        if self._evaluating:
            raise RuntimeError("Recursive evaluation is not supported.")
        if (
            self._evaluation_started
            or self._file_writer is not None
            or self.num_plays
            or getattr(self.renderer, "_target", None) is not None
            or getattr(self.renderer, "_context", None) is not None
        ):
            raise RuntimeError("No-raster evaluation requires a cold, unused Scene.")
        self._evaluation_started = True
        self._evaluating = True
        self.skip_animations = False
        try:
            self._check_evaluation_scene()
            self.setup()
            self._check_evaluation_scene()
            with contextlib.suppress(EndSceneEarlyException):
                self.construct()
            self.tear_down()
            self._check_evaluation_scene()
            self._finish_resource_scope()
        except BaseException:
            self._cleanup_after_failure()
            raise
        finally:
            self._evaluating = False

    def _check_evaluation_scene(self) -> None:
        if self._evaluating and self.scene.meshes:
            raise RuntimeError(
                "GPU-backed meshes are unsupported in no-raster evaluation."
            )

    def _finish_resource_scope(self) -> None:
        self._close_log_handler()
        if self._scope_depth == 0:
            self.close()

    def get_image(self) -> Image:
        """Draw a fresh scene image; see :meth:`.Scene.get_image` for snapshot behavior."""
        if self._evaluating:
            raise RuntimeError(
                "Image requests are unavailable during no-raster evaluation."
            )
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
        """Play animations and add an optional subcaption.

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
            Animation options such as ``run_time`` and ``rate_func``, applied
            by :meth:`.Scene.compile_animations`.
        """
        self._validate_execution()
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

    def _play(
        self, *args: Animation | Mobject | _AnimationBuilder, **kwargs: Any
    ) -> None:
        """Prepare and play animations, reusing cached frames when available."""
        self._validate_execution()
        scene = self.scene
        renderer: _AnimationRenderer = self.renderer
        self._check_evaluation_scene()
        self.skip_animations = (
            False if self._evaluating else self._execution.original_skipping_status
        )
        if not self._evaluating:
            self._update_skipping_status()
        scene.compile_animation_data(*args, **kwargs)
        if not self._evaluating:
            self._begin_animation_output()
        scene.begin_animations()
        self._check_evaluation_scene()
        if not self._evaluating:
            renderer._prepare_animation(scene)
        if scene.is_current_animation_frozen_frame():
            frame = None if self._evaluating else self._draw_animation_frame(0)
            frame_rate = self.session_spec.frame_rate
            repeats = int(scene.duration * frame_rate)
            if not self.skip_animations:
                self.time += repeats / frame_rate
                if frame is not None:
                    self.file_writer.write_frame(frame, repeat=repeats)
            if not self._evaluating:
                renderer._present_frozen_frame(scene, scene.duration)
        else:
            self._play_internal()
        if not self._evaluating:
            self.file_writer.end_animation(not self.skip_animations)
        self.num_plays += 1

    def _begin_animation_output(self) -> None:
        scene = self.scene
        renderer: _AnimationRenderer = self.renderer
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
                backend, drawing_state = renderer._animation_cache_identity(scene)
                hash_current_animation = get_hash_from_play_call(
                    scene,
                    self.camera,
                    scene.animations,
                    scene.mobjects,
                    backend=backend,
                    encoder_fingerprint=video_encoder_fingerprint(
                        self.session_spec.video_encoder
                    ),
                    renderer_state={
                        "drawing": drawing_state,
                        "execution": {
                            "clock": "sample-v2",
                            "time": self.time,
                            "play_index": self.num_plays,
                            "frame_rate": self.session_spec.frame_rate,
                        },
                    },
                )
                # Cached movies do not record when a stop condition became true.
                # Run these waits again to determine when they end.
                if scene.stop_condition is None and self.file_writer.is_already_cached(
                    hash_current_animation
                ):
                    logger.info(
                        f"Animation {self.num_plays} : Using cached data (hash : %(hash_current_animation)s)",
                        {"hash_current_animation": hash_current_animation},
                    )
                    self.skip_animations = True
                    self.time += self._sampled_duration(
                        scene.duration, scene.is_current_animation_frozen_frame()
                    )
        self.file_writer.add_partial_movie_file(hash_current_animation)
        self._execution.animations_hashes.append(hash_current_animation)
        logger.debug(
            "List of the first few animation hashes of the scene: %(h)s",
            {"h": str(self._execution.animations_hashes[:5])},
        )
        renderer._start_animation()
        self.file_writer.begin_animation(
            not self.skip_animations, animation_index=self.num_plays
        )

    def _sampled_duration(self, duration: float, frozen: bool) -> float:
        """Return the duration of the frames a normal render would produce."""
        frame_rate = self.session_spec.frame_rate
        count = (
            int(duration * frame_rate)
            if frozen
            else math.ceil(duration / (1 / frame_rate))
        )
        return count / frame_rate

    def _play_internal(self, skip_rendering: bool = False) -> None:
        """Step through animations, advancing time even when frames are not written."""
        self._validate_execution()
        scene = self.scene
        # Match Scene.get_time_progression and the scene's saved encoder settings.
        frame_rate = self.session_spec.frame_rate
        event_start = self.time
        assert scene.animations is not None
        scene.duration = scene.get_run_time(scene.animations)
        scene.time_progression = scene._get_animation_time_progression(
            scene.animations,
            scene.duration,
        )
        for sample_index, t in enumerate(scene.time_progression):
            scene.update_to_time(t)
            self._check_evaluation_scene()
            draw = (
                not self._evaluating
                and not skip_rendering
                and not scene.skip_animation_preview
            )
            frame = self._draw_animation_frame(t) if draw else None
            # Count the interval displayed by this frame, including the t=0 frame.
            # Stop conditions and finish() see the time at the end of that interval.
            if not self.skip_animations:
                self.time = event_start + (sample_index + 1) / frame_rate
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
        if self.skip_animations:
            return
        if frame is not None:
            self.file_writer.write_frame(frame)
        self.renderer._present_frame(self.scene, frame_offset)

    def _render_preview_frame(self, frame_offset: float) -> None:
        """Redraw and display an interactive frame without advancing animation time."""
        self._validate_execution()
        frame = self._draw_animation_frame(frame_offset)
        self._deliver_animation_frame(frame, frame_offset)

    def _legacy_add_frame(self, frame: RGBAPixelArray, num_frames: int = 1) -> None:
        """Write frames and advance time for callers of CairoRenderer.add_frame."""
        self._validate_execution()
        if not self.skip_animations:
            self.time += num_frames / self.session_spec.frame_rate
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
        if self._evaluating:
            return
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
        if self._evaluating:
            return
        subcaptions = self.file_writer.subcaptions
        subtitle = srt.Subtitle(
            index=len(subcaptions),
            content=content,
            start=datetime.timedelta(seconds=float(self.time + offset)),
            end=datetime.timedelta(seconds=float(self.time + offset + duration)),
        )
        subcaptions.append(subtitle)

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
        if self._evaluating:
            return
        self.file_writer.add_sound(sound_file, self.time + time_offset, gain, **kwargs)
