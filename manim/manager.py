"""Orchestration for rendering a scene."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from . import config, logger
from ._config.logger_utils import set_file_logger
from ._run.listeners import (
    _Encoder,
    _FrameRequests,
    _MeshGuard,
    _Preview,
    _Progress,
)
from ._run.loop import (
    _Caption,
    _RunListener,
    _RunLoop,
    _SceneImageDrawer,
    _Section,
    _Sound,
    _TargetDrawer,
)
from ._run.record import _RunRecord
from ._run.selection import _OutputSelection
from .scene.section import DefaultSectionType
from .utils.exceptions import EndSceneEarlyException, RerunSceneException
from .utils.file_ops import open_media_file

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
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
    from .timeline import Timeline
    from .typing import RGBAPixelArray

__all__ = ["Manager", "RenderedFrame"]

SceneT = TypeVar("SceneT", bound="Scene")


@dataclass(frozen=True)
class RenderedFrame:
    """A frame drawn during :meth:`.Manager.evaluate`.

    Parameters
    ----------
    image
        The frame as a PIL image in RGBA format.
    requested_time
        The requested time in seconds.
    time
        The start time of the frame in seconds.
    frame_index
        The position of the frame in the scene's frame sequence, counting from
        zero.
    """

    image: Image
    requested_time: float
    time: float
    frame_index: int


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
    before :meth:`~manim.scene.scene.Scene.setup` runs. :meth:`evaluate` runs the
    scene without producing media, and draws only the frames it is asked for.

    Both backends use the same animation clock and sampling rules. Every play
    steps through all of its frames, also when it is skipped or reused from the
    cache, so the scene always reaches the same state as in a full render.

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
        self._execution.frame_rate = self.session_spec.frame_rate
        self._run_loop: _RunLoop | None = None
        self._record: _RunRecord | None = None
        self._timeline: Timeline | None = None
        # The default of render(capture_timeline=...), set by the CLI.
        self._capture_timeline = False
        self._evaluating = False
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
        if self._evaluating or self._log_handler is not None:
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
        """The renderer selected when the scene was created."""
        return self.scene.renderer

    @property
    def camera(self) -> Camera | OpenGLCamera:
        """The current renderer's camera."""
        return self.renderer.camera

    @property
    def file_writer(self) -> SceneFileWriter:
        """The scene's file writer, created on first access.

        The writer uses ``scene.file_writer_settings``. :meth:`render` accesses
        this property before calling :meth:`setup`; ``renderer.file_writer``
        returns the same writer. Both properties are read-only. To choose a custom
        writer class, pass ``file_writer_class`` when constructing the renderer.
        Accessing this property during :meth:`evaluate` raises ``RuntimeError``.
        """
        self._ensure_not_evaluating("Media output")
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
        """The output format and options saved for this scene."""
        return self.session_spec.output

    @property
    def session_spec(self) -> RenderSessionSpec:
        """The scene's saved output, preview, and execution settings."""
        return self.scene.session_spec

    @property
    def time(self) -> float:
        """The elapsed animation time in seconds."""
        return self._execution.time

    @time.setter
    def time(self, value: float) -> None:
        self._execution.time = value

    @property
    def num_plays(self) -> int:
        """The number of completed play or wait calls."""
        return self._execution.num_plays

    @num_plays.setter
    def num_plays(self, value: int) -> None:
        self._execution.num_plays = value

    @property
    def skip_animations(self) -> bool:
        """Whether the current play draws no frames.

        This is the case for a play that is left out of the output or reused
        from the cache. The play still steps through all of its frames.
        """
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

    def render(
        self, preview: bool = False, *, capture_timeline: bool | None = None
    ) -> bool:
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
        capture_timeline
            Whether to record the scene's play, wait, section, caption, and sound
            calls. After a successful render, the result is available as
            :attr:`timeline`.

        Returns
        -------
        bool
            ``True`` when an interactive rerun was requested; otherwise
            ``False``. This matches the return value of
            :meth:`~manim.scene.scene.Scene.render`.
        """
        from .renderer.opengl.renderer import OpenGLRenderer

        self._ensure_not_evaluating("Rendering")
        self._validate_execution()
        started = False
        try:
            presentation = self.session_spec.presentation
            open_after_render = preview or presentation.open_after_render
            if open_after_render and not self.output_spec.enabled:
                raise ValueError("Previewing after render requires a media artifact.")

            started = True
            self._timeline = None
            record = self._loop_record()
            if capture_timeline is None:
                capture_timeline = self._capture_timeline
            record.capture = capture_timeline
            self._open_log_handler()
            if isinstance(self.renderer, OpenGLRenderer):
                self.renderer.open()
            # Make the writer available to user setup() code.
            _ = self.file_writer
            self.setup()
            ended_early = False
            try:
                self.construct()
            except EndSceneEarlyException:
                # Only a construction boundary is a normal early scene end.
                ended_early = True
            except RerunSceneException:
                # The caller will create a fresh Scene for the rerun.
                # Report any encoder errors before allowing that restart.
                self.file_writer.abort_encode_jobs(reraise_encoder_failures=True)
                self._finish_resource_scope()
                return True
            self.tear_down()
            self._loop.finish(ended_early=ended_early)
            self.post_construct()
            if capture_timeline:
                self._timeline = record.timeline()

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

    @property
    def timeline(self) -> Timeline:
        """The timeline recorded by a successful call to :meth:`evaluate` or
        :meth:`render` with ``capture_timeline=True``.

        Raises ``RuntimeError`` if no timeline has been recorded.
        """
        if self._timeline is None:
            raise RuntimeError("No completed timeline capture is available.")
        return self._timeline

    def evaluate(
        self,
        *,
        capture_timeline: bool = False,
        frame_times: Iterable[float] = (),
        on_frame: Callable[[RenderedFrame], object] | None = None,
    ) -> None:
        """Run the scene's animations without producing media.

        Calls ``setup()``, ``construct()`` and ``tear_down()``, using the same
        animation steps and clock as rendering. Movie caches, animation ranges,
        and skip flags are ignored, so every play runs. Only the frames requested
        with ``frame_times`` are drawn. Section, subcaption, and sound calls have
        no effect; sound files are not checked.

        Parameters
        ----------
        capture_timeline
            Whether to record the scene's play, wait, section, caption, and sound
            calls. After a successful evaluation, the result is available as
            :attr:`timeline`.
        frame_times
            Times in seconds at which to draw the frame that a render would show.
            Each frame is drawn when evaluation reaches it and passed to
            ``on_frame``. Times at or after the end of the scene are ignored.
        on_frame
            Called with a :class:`RenderedFrame` for each time in
            ``frame_times``, in increasing order of time. It runs inside the
            scene's ``play()`` or ``wait()`` call, so an exception it raises ends
            the evaluation.

        Notes
        -----
        The scene must be unused: it must not have played animations or rendered
        yet. Each scene can be evaluated once. Keep the frame-rate configuration
        used to construct it.

        Afterward, inspect ``scene.time`` and the scene's mobjects, or call
        ``scene.get_image()`` to draw the resulting state; it can also be called
        during evaluation. The file writer, nested renders, and interactive
        preview raise an error during evaluation.

        The manager closes on success, or at the end of an enclosing
        ``with manager:`` block. If evaluation fails, the manager cleans up and
        raises the original exception.

        See :doc:`/guides/evaluation` for examples.
        """
        self._validate_execution()
        if self._evaluating:
            raise RuntimeError("Recursive evaluation is not supported.")
        if (
            self._run_loop is not None
            or self._file_writer is not None
            or self.num_plays
            or getattr(self.renderer, "_target", None) is not None
            or getattr(self.renderer, "_context", None) is not None
        ):
            raise RuntimeError("Evaluation requires an unused Scene.")
        times = [float(time) for time in frame_times]
        if any(not math.isfinite(time) or time < 0 for time in times):
            raise ValueError("Frame times must be finite and not negative.")
        if times and on_frame is None:
            raise ValueError("Pass on_frame to receive the frames of frame_times.")
        record = self._record = _RunRecord(self, "evaluate")
        record.capture = capture_timeline
        meshes = _MeshGuard(self)
        listeners: list[_RunListener] = [record, meshes, _Progress(self)]
        if on_frame is not None:
            listeners.append(_FrameRequests(self, times, on_frame))
        self._run_loop = _RunLoop(
            self,
            listeners=listeners,
            selection=_OutputSelection.everything(self),
            drawer=_SceneImageDrawer(self),
        )
        self._evaluating = True
        try:
            meshes.check()
            self.setup()
            ended_early = False
            try:
                self.construct()
            except EndSceneEarlyException:
                ended_early = True
            self.tear_down()
            meshes.check()
            self._run_loop.finish(ended_early=ended_early)
            if capture_timeline:
                self._timeline = record.timeline()
            self._finish_resource_scope()
        except BaseException:
            self._cleanup_after_failure()
            raise
        finally:
            self._evaluating = False

    def _ensure_not_evaluating(self, what: str) -> None:
        if self._evaluating:
            raise RuntimeError(f"{what} is unavailable during evaluation.")

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

    def _loop_record(self) -> _RunRecord:
        _ = self._loop
        assert self._record is not None
        return self._record

    @property
    def _loop(self) -> _RunLoop:
        """The loop that runs this scene's play calls, created on first use."""
        if self._run_loop is None:
            self._record = _RunRecord(self, "render")
            self._run_loop = _RunLoop(
                self,
                listeners=[
                    self._record,
                    _Encoder(self),
                    _Preview(self),
                    _Progress(self),
                ],
                selection=_OutputSelection(self),
                drawer=_TargetDrawer(self),
            )
        return self._run_loop

    def _update_skipping_status(self) -> None:
        """Set :attr:`skip_animations` for the next play, for old renderer callers.

        Raises
        ------
        EndSceneEarlyException
            If the next play lies after ``config.upto_animation_number``.
        """
        self.skip_animations = not self._loop.selection.start_play(self.num_plays)

    def _play(
        self, *args: Animation | Mobject | _AnimationBuilder, **kwargs: Any
    ) -> None:
        """Prepare and play animations, reusing cached frames when available."""
        self._validate_execution()
        self._loop.play(*args, **kwargs)

    def _wants_frame_pixels(self) -> bool:
        """Whether the file writer is given the pixels of drawn frames.

        Reading pixels back is cheap for Cairo, and test writers record them
        even without file output. OpenGL reads them back only for video and
        image-sequence output.
        """
        from .renderer.cairo.renderer import CairoRenderer

        output = self.output_spec
        return (
            isinstance(self.renderer, CairoRenderer)
            or output.is_video
            or output.is_image_sequence
        )

    def _render_preview_frame(self, frame_offset: float) -> None:
        """Draw and show the current state without advancing animation time.

        Interactive OpenGL sessions use this between commands. Unless the current
        play is skipped, the frame is also passed to the file writer.
        """
        self._validate_execution()
        scene = self.scene
        self.renderer.render(scene, frame_offset, scene.moving_mobjects)
        if self.skip_animations:
            return
        if self._wants_frame_pixels():
            self.file_writer.write_frame(self.renderer.get_frame())
        self.renderer._present_frame(scene, frame_offset)

    def _legacy_add_frame(self, frame: RGBAPixelArray, num_frames: int = 1) -> None:
        """Write frames and advance time for callers of CairoRenderer.add_frame."""
        self._validate_execution()
        if not self.skip_animations:
            self._execution.frame += num_frames
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
        self._loop.marker(_Section(name, section_type, skip_animations))

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
        start = float(self.time + offset)
        self._loop.marker(_Caption(content, start, start + float(duration)))

    def add_sound(
        self,
        sound_file: str,
        time_offset: float = 0,
        gain: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Add sound to the output at the current scene time.

        No sound is added while the surrounding plays are left out of the output.

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
        self._loop.marker(
            _Sound(
                sound_file,
                self.time + time_offset,
                gain,
                dict(kwargs),
                in_output=self._loop.in_output,
            )
        )
