"""Orchestration for rendering a scene."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import srt

from . import logger
from ._config.logger_utils import set_file_logger
from .scene.section import DefaultSectionType
from .utils.exceptions import EndSceneEarlyException, RerunSceneException
from .utils.file_ops import open_media_file

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

__all__ = ["Manager"]

SceneT = TypeVar("SceneT", bound="Scene")


class Manager(Generic[SceneT]):
    """Coordinate the render lifecycle for a single scene.

    A manager is attached to exactly one :class:`~manim.scene.scene.Scene`. It
    coordinates the scene lifecycle, delegates animation playback to the current
    renderer, and routes section, subcaption, and audio operations to the current
    file writer. Calling :meth:`~manim.scene.scene.Scene.render` creates a manager
    lazily when one has not already been attached.

    Parameters
    ----------
    scene
        The scene coordinated by this manager. A scene that already has a manager
        cannot be attached to another one.

    Attributes
    ----------
    scene
        The managed scene.

    Notes
    -----
    This class is the coordination boundary for an incremental render-flow
    refactor. The Manager creates and owns the file writer on first output demand.
    Renderers retain a compatibility view for their legacy schedulers. Renderer,
    camera, and clock ownership remain unchanged; output settings are still
    resolved during Scene construction.

    Examples
    --------
    A scene normally creates its manager when :meth:`~manim.scene.scene.Scene.render`
    is called. It can also be attached explicitly::

        scene = Scene()
        manager = Manager(scene)
        manager.render()
    """

    def __init__(self, scene: SceneT) -> None:
        if scene.manager is not None:
            raise ValueError("A manager is already attached to this scene.")
        self.scene = scene
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
        """Drain output, retire the backend, and detach only this Manager's log.

        Successful render retires resources automatically unless an explicit
        Manager context scope extends their lifetime for inspection.
        """
        if self._closed:
            return
        self._closing = True
        failures: list[BaseException] = []

        def cleanup(callback: Callable[[], Any]) -> None:
            try:
                callback()
            except BaseException as error:
                failures.append(error)

        if self._file_writer is not None:
            cleanup(self._file_writer.abort_encode_jobs)

        def close_backend() -> None:
            # A legacy rebind transfers the backend to another Scene, not its writer.
            if self.renderer._is_bound_to(self.scene):
                self.renderer.close()

        cleanup(close_backend)
        cleanup(self._close_log_handler)
        if failures:
            for secondary in failures[1:]:
                logger.error("Additional Manager cleanup failure", exc_info=secondary)
            raise failures[0]
        self._closed = True

    def _cleanup_after_failure(self) -> None:
        try:
            self.close()
        except BaseException:
            logger.exception("Failed to clean up resources after a render failure")

    @property
    def renderer(self) -> CairoRenderer | OpenGLRenderer:
        """Return the scene's current renderer."""
        return self.scene.renderer

    @property
    def camera(self) -> Camera | OpenGLCamera:
        """Return the current renderer's camera."""
        return self.renderer.camera

    @property
    def file_writer(self) -> SceneFileWriter:
        """Return the owned writer, creating it on first output demand.

        Image inspection does not request a writer. Explicit legacy access through
        ``renderer.file_writer`` does, and attaches a Manager if necessary.
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

    def _replace_file_writer(self, writer: SceneFileWriter) -> None:
        if self._closed or self._closing or self._creating_file_writer:
            raise RuntimeError(
                "Cannot replace output resources in this lifecycle state."
            )
        previous = self._file_writer
        if previous is writer:
            return
        if previous is not None:
            previous.abort_encode_jobs(reraise_encoder_failures=True)
        self._file_writer = writer

    @property
    def output_spec(self) -> OutputSpec:
        """Return the immutable output intent captured for this session."""
        return self.session_spec.output

    @property
    def session_spec(self) -> RenderSessionSpec:
        """Return the immutable artifact, presentation, and execution intent."""
        return self.scene.session_spec

    @property
    def time(self) -> float:
        """Return the current renderer time."""
        return self.renderer.time

    @time.setter
    def time(self, value: float) -> None:
        self.renderer.time = value

    @property
    def num_plays(self) -> int:
        """Return the current renderer's play count."""
        return self.renderer.num_plays

    @num_plays.setter
    def num_plays(self, value: int) -> None:
        self.renderer.num_plays = value

    @property
    def skip_animations(self) -> bool:
        """Return the current renderer's animation skip state."""
        return self.renderer.skip_animations

    @skip_animations.setter
    def skip_animations(self, value: bool) -> None:
        self.renderer.skip_animations = value

    def render(self, preview: bool = False) -> bool:
        """Run the complete render lifecycle for the managed scene.

        The lifecycle invokes :meth:`setup`, :meth:`construct`,
        :meth:`tear_down`, and :meth:`post_construct`, in that order. Reaching a
        configured animation boundary ends construction normally. Failures during
        setup, construction, teardown, finalization, or preview opening abort
        encoding jobs before the original exception is propagated. Successful
        execution retires resources before returning, unless an explicit Manager
        context scope extends the backend's inspection lifetime to scope exit.

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
            # Preserve writer availability in user setup without opening it
            # during Scene construction or non-output image inspection.
            _ = self.file_writer
            self.setup()
            try:
                self.construct()
            except EndSceneEarlyException:
                # Only a construction boundary is a normal early scene end.
                pass
            except RerunSceneException:
                self.scene.remove(*self.scene.mobjects)
                # TODO: The CairoRenderer does not have the method clear_screen().
                self.renderer.clear_screen()  # type: ignore[union-attr]
                self.num_plays = 0
                # A rerun has no primary failure to preserve: encoder failures
                # must prevent reuse of an incomplete/corrupt output session.
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
            # Even setup or teardown can leave a non-daemon encoder waiting for
            # frames. Cleanup must not replace the exception that brought us here.
            if started or self._file_writer is not None:
                self._cleanup_after_failure()
            raise

    def _finish_resource_scope(self) -> None:
        self._close_log_handler()
        if self._scope_depth == 0:
            self.close()

    def get_image(self) -> Image:
        """Materialize current state without entering a timed/output transaction."""
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
        and logs the number of played animations. It intentionally runs after
        :meth:`tear_down` to preserve the established render lifecycle.
        """
        output = self.output_spec
        empty_video_output = self.num_plays == 0 and output.is_video
        if empty_video_output and not output.fallback_to_still:
            raise RuntimeError(
                f"{self.scene} has no play calls, so the explicitly requested "
                f"{output.format.value.upper()} output cannot be produced. "
                "Use --format=png to save its last frame.",
            )

        # We have to reset these settings in case of multiple renders.
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
            Additional animation arguments forwarded to the renderer.
        """
        if self._closed or self._closing:
            raise RuntimeError("The Manager is closed or closing.")
        self._open_log_handler()
        start_time = self.time
        self.renderer.play(self.scene, *args, **kwargs)
        run_time = self.time - start_time

        if subcaption:
            if subcaption_duration is None:
                subcaption_duration = run_time
            # The start of the subcaption needs to be offset by the run time
            # because it is added after the animation has already played.
            # Route through Scene's public API to preserve its customization hook.
            self.scene.add_subcaption(
                content=subcaption,
                duration=subcaption_duration,
                offset=-run_time + subcaption_offset,
            )

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
