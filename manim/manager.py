"""Orchestration for rendering a scene."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import srt

from . import logger
from .scene.section import DefaultSectionType
from .utils.exceptions import EndSceneEarlyException, RerunSceneException
from .utils.file_ops import open_media_file

if TYPE_CHECKING:
    from ._config.output import OutputSpec
    from ._config.render_session import RenderSessionSpec
    from .animation.animation import Animation
    from .camera.camera import Camera
    from .mobject.mobject import Mobject, _AnimationBuilder
    from .renderer.cairo_renderer import CairoRenderer
    from .renderer.opengl_renderer import OpenGLCamera, OpenGLRenderer
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
    refactor. Renderer, camera, clock, and output ownership still remain on the
    scene and renderer for compatibility. The :attr:`renderer`, :attr:`camera`,
    :attr:`file_writer`, :attr:`time`, :attr:`num_plays`, and
    :attr:`skip_animations` properties are forwarding views for now.

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
        scene.manager = self

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
        """Return the current renderer's file writer."""
        return cast("SceneFileWriter", self.renderer.file_writer)

    @property
    def output_spec(self) -> OutputSpec:
        """Return the immutable output intent captured for this session."""
        return self.session_spec.output

    @property
    def session_spec(self) -> RenderSessionSpec:
        """Return the immutable output and presentation intent for this session."""
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
        configured animation boundary ends construction normally. Exceptions
        raised while constructing the scene abort active encoding jobs before
        they are propagated.

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
        presentation = self.session_spec.presentation
        open_after_render = preview or presentation.open_after_render
        if open_after_render and not self.output_spec.enabled:
            raise ValueError("Previewing after render requires a media artifact.")

        self.setup()
        try:
            self.construct()
        except EndSceneEarlyException:
            # Reaching the configured animation boundary ends the scene normally.
            pass
        except RerunSceneException:
            self.scene.remove(*self.scene.mobjects)
            # TODO: The CairoRenderer does not have the method clear_screen().
            self.renderer.clear_screen()  # type: ignore[union-attr]
            self.num_plays = 0
            # The rerun replaces the file writer; tear down its encode jobs so
            # no worker is still writing a partial file the new writer may
            # reuse. Encoder failures propagate: a rerun must not silently
            # continue past corrupt output.
            self.file_writer.abort_encode_jobs(reraise_encoder_failures=True)
            return True
        except BaseException:
            # A mid-play exception leaves an unsealed encode job whose
            # non-daemon worker would hang the process at exit.
            self.file_writer.abort_encode_jobs()
            raise
        self.tear_down()
        self.post_construct()

        if open_after_render or presentation.show_in_file_browser:
            open_media_file(
                self.file_writer,
                preview=open_after_render,
                show_in_file_browser=presentation.show_in_file_browser,
            )

        return False

    def setup(self) -> None:
        """Run the managed scene's :meth:`~manim.scene.scene.Scene.setup` hook."""
        self.scene.setup()

    def construct(self) -> None:
        """Run the managed scene's :meth:`~manim.scene.scene.Scene.construct` hook."""
        self.scene.construct()

    def post_construct(self) -> None:
        """Finalize output after scene construction and tear-down.

        This asks the renderer to finish the scene and logs the number of played
        animations. It intentionally runs after :meth:`tear_down` to preserve the
        established render lifecycle.
        """
        # We have to reset these settings in case of multiple renders.
        self.renderer.scene_finished(self.scene)

        # Show info only if animations are rendered or to get image.
        if self.num_plays or self.output_spec.enabled:
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
