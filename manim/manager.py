"""Orchestration for rendering a scene."""

from __future__ import annotations

import datetime
import math
from bisect import bisect_right
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import srt
from PIL import Image as PILImage

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
    from .timeline import Timeline, _SourceSnapshot, _TimelineRecorder
    from .typing import RGBAPixelArray

__all__ = ["Manager", "RenderedFrame"]

SceneT = TypeVar("SceneT", bound="Scene")


@dataclass(frozen=True)
class RenderedFrame:
    """A frame captured by :meth:`.Manager.capture_frame_at`.

    Parameters
    ----------
    image
        An independent PIL image in RGBA format.
    requested_time
        The requested timestamp in seconds.
    time
        The selected frame's start time in seconds.
    frame_index
        The selected frame's position, counting from zero.
    """

    image: Image
    requested_time: float
    time: float
    frame_index: int


@dataclass
class _FrameRequest:
    timestamp: float
    frames: int = 0
    result: RenderedFrame | None = None
    stopped: bool = False
    playing: bool = False


class _FrameCaptured(BaseException):
    """Stop scene execution after capturing the requested frame."""


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
    Output settings are saved during scene construction. During rendering, the
    manager creates the file writer when first requested and makes it available
    before :meth:`~manim.scene.scene.Scene.setup` runs. Use :meth:`evaluate` to
    inspect the scene's animation time and mobject state, or :meth:`capture_frame_at`
    to get an image at a requested timestamp.

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
        self._frame_request: _FrameRequest | None = None
        self._timeline: Timeline | None = None
        self._timeline_recorder: _TimelineRecorder | None = None
        self._timeline_source: _SourceSnapshot | None = None
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
        if self._evaluating or self._frame_request is not None:
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
        Accessing this property during :meth:`evaluate` or :meth:`capture_frame_at`
        raises ``RuntimeError``.
        """
        if self._evaluating:
            raise RuntimeError(
                "Media output is unavailable during no-raster evaluation."
            )
        if self._frame_request is not None:
            raise RuntimeError("Media output is unavailable during frame capture.")
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
        if self._frame_request is not None and self._frame_request.stopped:
            raise RuntimeError(
                "Frame capture has ended; use a fresh Scene for further playback."
            )
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
        if self._frame_request is not None:
            raise RuntimeError("Frame capture requires its own Scene execution.")
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

    def capture_frame_at(self, timestamp: float) -> RenderedFrame | None:
        """Return the frame displayed at ``timestamp`` by executing a fresh scene.

        Runs the scene's animation steps from the beginning, drawing each frame
        with Cairo or OpenGL until the requested frame is reached. The returned
        image can be viewed or saved after the renderer closes. For an image of
        the scene's current state, use :meth:`get_image`.

        Parameters
        ----------
        timestamp
            A finite time of zero or greater, in seconds from the start of the
            frame sequence. At 4 fps, times from 0.25 up to but excluding 0.5
            select the frame starting at 0.25. During a frozen wait, capture
            returns the held image.

        Returns
        -------
        RenderedFrame | None
            The image, its frame index, and its start time. Returns ``None`` for
            a timestamp at or beyond the end of the frame sequence.

        Notes
        -----
        Create a fresh scene with ``live_preview=False`` for each request, and
        keep the frame rate chosen at construction. ``setup()`` and ``construct()``
        run until capture. Earlier animations finish normally; playback stops
        with the selected animation at its captured state. Python ``finally``
        blocks run, followed by ``tear_down()`` for cleanup. The image retains
        the captured pixels through cleanup.

        A successful call closes the renderer, or keeps it open until an enclosing
        ``with manager:`` block ends. If scene code fails, Manager attempts resource
        cleanup and raises the original error. For OpenGL, make the call on the
        thread that opened the scene's context.

        Examples
        --------
        At 4 fps, a request at 0.3 seconds selects the frame starting at 0.25::

            class Motion(Scene):
                def construct(self):
                    square = Square()
                    self.add(square)
                    self.play(square.animate.shift(RIGHT))


            with tempconfig({"frame_rate": 4, "live_preview": False}):
                scene = Motion()
                manager = Manager(scene)
                frame = manager.capture_frame_at(0.3)
                if frame is not None:
                    print(frame.time)  # 0.25
                    frame.image.save("frame.png")
        """
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and nonnegative.")
        self._validate_execution()
        if (
            self._frame_request is not None
            or self._evaluation_started
            or self._file_writer is not None
            or self.num_plays
            or self.time != 0
        ):
            raise RuntimeError("Frame capture requires a fresh, unused Scene.")
        if self.session_spec.presentation.live_preview:
            raise ValueError(
                "Construct the Scene with live_preview=False for frame capture."
            )
        request = self._frame_request = _FrameRequest(timestamp)
        try:
            self.renderer._start_animation()
            with suppress(_FrameCaptured):
                self.setup()
                with suppress(EndSceneEarlyException):
                    self.construct()
            request.stopped = True
            self.tear_down()
            self._finish_resource_scope()
            return request.result
        except BaseException:
            self._cleanup_after_failure()
            raise

    @property
    def timeline(self) -> Timeline:
        """Return the completed timeline requested during :meth:`evaluate`.

        Read this property after ``evaluate(capture_timeline=True)`` returns
        successfully. The returned snapshot is immutable.
        """
        if self._timeline is None:
            raise RuntimeError("No completed timeline capture is available.")
        return self._timeline

    def evaluate(self, *, capture_timeline: bool = False) -> None:
        """Run the scene's animations without drawing frames or producing media.

        Calls ``setup()``, ``construct()`` and ``tear_down()``, using the same
        animation steps and clock as uncached, unskipped rendering. Movie caches,
        animation ranges, and skip flags are ignored. Section, subcaption, and
        sound calls produce no media output; sound files are not checked. Optional
        timeline capture records these calls and summarizes each play or wait.

        Parameters
        ----------
        capture_timeline
            Record play/wait timing, step counts, and section/caption/sound calls
            in :attr:`timeline` after successful evaluation. Use
            :meth:`.Timeline.write` to save the captured report as JSON.

        Notes
        -----
        Start with a fresh scene, before playing animations or opening its
        renderer's drawing resources or file writer. Each scene can be evaluated
        once. Keep the frame-rate configuration used to construct it.

        Afterward, inspect ``scene.time`` and the scene's mobjects. An explicit
        ``scene.get_image()`` call can then draw the resulting state. Image,
        renderer GPU, writer, and interactive-preview requests during evaluation
        raise an error instead.

        The manager closes on success, or at the end of an enclosing
        ``with manager:`` block. Failures trigger cleanup and preserve the original
        exception. No file writer, encoder, preview window, or file log is opened
        by evaluation. User code still runs and can perform its own I/O: this is
        not a sandbox.
        """
        self._validate_execution()
        if self._frame_request is not None:
            raise RuntimeError("Frame capture requires its own Scene execution.")
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
            if capture_timeline:
                from .timeline import _TimelineRecorder

                self._timeline_recorder = _TimelineRecorder(
                    self.scene,
                    self.session_spec.frame_rate,
                    self.time,
                    self._timeline_source,
                )
            self._check_evaluation_scene()
            self.setup()
            self._check_evaluation_scene()
            try:
                self.construct()
            except EndSceneEarlyException:
                if self._timeline_recorder is not None:
                    self._timeline_recorder.termination = "scene-end-request"
            self.tear_down()
            self._check_evaluation_scene()
            self._finish_resource_scope()
            if self._timeline_recorder is not None:
                self._timeline = self._timeline_recorder.finish(self.time)
        except BaseException:
            self._cleanup_after_failure()
            raise
        finally:
            self._evaluating = False
            self._timeline_recorder = None

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
        """Return an image of the scene's current mobjects and camera view.

        See :meth:`.Scene.get_image` for snapshot usage. To execute a fresh scene
        up to a chosen timestamp, use :meth:`capture_frame_at`.
        """
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
        request = self._frame_request
        if request is not None:
            if request.playing:
                raise RuntimeError(
                    "Recursive play/wait calls are unsupported during frame capture."
                )
            request.playing = True
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
        except (_FrameCaptured, EndSceneEarlyException, RerunSceneException):
            # Let the execution entry point handle a requested stop or rerun.
            raise
        except BaseException:
            # A direct play() can fail outside render(), so it needs cleanup here too.
            self._cleanup_after_failure()
            raise
        finally:
            if request is not None:
                request.playing = False

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
        """Prepare and play animations, checking the movie cache only when rendering."""
        self._validate_execution()
        scene = self.scene
        renderer: _AnimationRenderer = self.renderer
        self._check_evaluation_scene()
        writing = not self._evaluating and self._frame_request is None
        self.skip_animations = (
            self._execution.original_skipping_status if writing else False
        )
        if writing:
            self._update_skipping_status()
        event_start = self.time
        event_ordinal = self.num_plays
        if self._timeline_recorder is not None:
            self._timeline_recorder.enter(event_start)
        scene.compile_animation_data(*args, **kwargs)
        if self._timeline_recorder is not None:
            self._timeline_recorder.begin(scene, event_start, event_ordinal)
        if writing:
            self._begin_animation_output()
        elif not self._evaluating:
            renderer._start_animation()
        scene.begin_animations()
        self._check_evaluation_scene()
        if not self._evaluating:
            renderer._prepare_animation(scene)
        if scene.is_current_animation_frozen_frame():
            frame = None if self._evaluating else self._draw_animation_frame(0)
            frame_rate = self.session_spec.frame_rate
            repeats = int(scene.duration * frame_rate)
            if not self.skip_animations:
                start_time = self.time
                self.time += repeats / frame_rate
                if self._timeline_recorder is not None:
                    self._timeline_recorder.hold(repeats, self.time)
                if frame is not None:
                    self._write_animation_frame(
                        frame, repeat=repeats, start_time=start_time
                    )
            if writing:
                renderer._present_frozen_frame(scene, scene.duration)
        else:
            self._play_internal()
        if writing:
            self.file_writer.end_animation(not self.skip_animations)
        self.num_plays += 1
        if self._timeline_recorder is not None:
            self._timeline_recorder.end(self.time)

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
        try:
            for sample_index, t in enumerate(scene.time_progression):
                scene.update_to_time(t)
                self._check_evaluation_scene()
                draw = (
                    not self._evaluating
                    and not skip_rendering
                    and not scene.skip_animation_preview
                )
                frame = self._draw_animation_frame(t) if draw else None
                # Count this step's frame interval even when evaluation draws no frame.
                # Stop conditions and finish() see the time at the end of that interval.
                if not self.skip_animations:
                    self.time = event_start + (sample_index + 1) / frame_rate
                    if self._timeline_recorder is not None:
                        self._timeline_recorder.sample(self.time)
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
        finally:
            self.renderer.static_image = None  # type: ignore[union-attr]
            scene.time_progression.close()

    def _draw_animation_frame(self, frame_offset: float) -> RGBAPixelArray | None:
        from .renderer.cairo.renderer import CairoRenderer

        self.renderer.render(self.scene, frame_offset, self.scene.moving_mobjects)
        if (
            self._frame_request is not None
            or isinstance(self.renderer, CairoRenderer)
            or (
                not self.skip_animations
                and (
                    self.file_writer.output_spec.is_video
                    or self.file_writer.output_spec.is_image_sequence
                )
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
            self._write_animation_frame(frame)
        if self._frame_request is None:
            self.renderer._present_frame(self.scene, frame_offset)

    def _write_animation_frame(
        self, frame: RGBAPixelArray, *, repeat: int = 1, start_time: float | None = None
    ) -> None:
        request = self._frame_request
        if request is None:
            self.file_writer.write_frame(frame, repeat=repeat)
            return
        if request.stopped:
            raise RuntimeError("Animation playback has ended for this frame request.")
        first = request.frames
        request.frames += repeat
        frame_rate = self.session_spec.frame_rate
        if request.timestamp < request.frames / frame_rate:
            # Compare interval boundaries directly, including within a frozen hold.
            # Multiplying the timestamp by fps can round across an exact boundary.
            index = (
                first
                + bisect_right(
                    range(first, request.frames),
                    request.timestamp,
                    key=lambda i: i / frame_rate,
                )
                - 1
            )
            if start_time is not None:
                self.time = start_time + (index - first + 1) / frame_rate
            request.result = RenderedFrame(
                image=PILImage.fromarray(frame).copy(),
                requested_time=request.timestamp,
                time=index / frame_rate,
                frame_index=index,
            )
            request.stopped = True
            raise _FrameCaptured

    def _render_preview_frame(self, frame_offset: float) -> None:
        """Redraw and display an interactive frame without advancing animation time."""
        self._validate_execution()
        frame = self._draw_animation_frame(frame_offset)
        self._deliver_animation_frame(frame, frame_offset)

    def _legacy_add_frame(self, frame: RGBAPixelArray, num_frames: int = 1) -> None:
        """Write frames and advance time for callers of CairoRenderer.add_frame."""
        self._validate_execution()
        if not self.skip_animations:
            start_time = self.time
            self.time += num_frames / self.session_spec.frame_rate
            self._write_animation_frame(frame, repeat=num_frames, start_time=start_time)

    def next_section(
        self,
        name: str = "unnamed",
        section_type: str = DefaultSectionType.NORMAL,
        skip_animations: bool = False,
    ) -> None:
        """Create a new output section.

        During :meth:`evaluate`, this call produces no output. When timeline
        capture is enabled, it is recorded in the timeline instead.

        Parameters
        ----------
        name
            The section name.
        section_type
            The section type stored in the section manifest.
        skip_animations
            Whether animation output in this section should be skipped.
        """
        if self._frame_request is not None:
            return
        if self._evaluating:
            if self._timeline_recorder is not None:
                self._timeline_recorder.declare(
                    "section",
                    self.time,
                    self.num_plays,
                    name=name,
                    type=section_type,
                    skip_requested=skip_animations,
                )
            return
        self.file_writer.next_section(name, section_type, skip_animations)

    def add_subcaption(
        self, content: str, duration: float = 1, offset: float = 0
    ) -> None:
        """Add a subcaption at the current scene time.

        During :meth:`evaluate`, this call produces no output. When timeline
        capture is enabled, the caption is recorded in the timeline instead.

        Parameters
        ----------
        content
            The subcaption text.
        duration
            The duration in seconds for which the subcaption is displayed.
        offset
            The offset in seconds from the current scene time.
        """
        if self._frame_request is not None or (
            self._evaluating and self._timeline_recorder is None
        ):
            return
        start = datetime.timedelta(seconds=float(self.time + offset))
        end = datetime.timedelta(seconds=float(self.time + offset + duration))
        if not self._evaluating:
            subcaptions = self.file_writer.subcaptions
            subcaptions.append(
                srt.Subtitle(
                    index=len(subcaptions), content=content, start=start, end=end
                )
            )
        if self._timeline_recorder is not None:
            self._timeline_recorder.declare(
                "caption",
                self.time,
                self.num_plays,
                content=content,
                start=start.total_seconds(),
                end=end.total_seconds(),
            )

    def add_sound(
        self,
        sound_file: str,
        time_offset: float = 0,
        gain: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Add sound to the output at the current scene time.

        No sound is added while animations are being skipped. During
        :meth:`evaluate`, this call also produces no output; the sound file is
        neither checked nor decoded. When timeline capture is enabled, the sound
        request is recorded in the timeline instead.

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
        if self.skip_animations or self._frame_request is not None:
            return
        if self._evaluating:
            if self._timeline_recorder is not None:
                self._timeline_recorder.declare(
                    "sound",
                    self.time,
                    self.num_plays,
                    asset=self._timeline_recorder.asset_hint(sound_file),
                    start=self.time + time_offset,
                    gain=gain,
                    options=dict(kwargs),
                    duration=None,
                )
            return
        self.file_writer.add_sound(sound_file, self.time + time_offset, gain, **kwargs)
