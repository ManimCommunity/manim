"""Orchestration for rendering a scene."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import srt

from . import config, logger
from .scene.section import DefaultSectionType
from .utils.exceptions import EndSceneEarlyException, RerunSceneException
from .utils.file_ops import open_media_file

if TYPE_CHECKING:
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
    """Coordinate the lifecycle and renderer calls for a single scene.

    This is deliberately a concrete orchestration object.  Renderer, camera, and
    output ownership remain on :class:`~manim.scene.scene.Scene` and its renderer
    for now.
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
        """Render the manager's scene.

        Returns ``True`` when an interactive rerun was requested, matching the
        historical :meth:`Scene.render` return value.
        """
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
            return True
        self.tear_down()
        self.post_construct()

        # If preview open up the render after rendering.
        if preview:
            config["preview"] = True

        if config["preview"] or config["show_in_file_browser"]:
            open_media_file(self.file_writer)

        return False

    def setup(self) -> None:
        """Run the scene's setup hook."""
        self.scene.setup()

    def construct(self) -> None:
        """Run the scene's construct hook."""
        self.scene.construct()

    def post_construct(self) -> None:
        """Finalize output after the scene's construction has completed.

        This intentionally runs after :meth:`tear_down` to preserve main's
        established render lifecycle.
        """
        # We have to reset these settings in case of multiple renders.
        self.renderer.scene_finished(self.scene)

        # Show info only if animations are rendered or to get image.
        if self.num_plays or config["format"] == "png" or config["save_last_frame"]:
            logger.info(
                f"Rendered {str(self.scene)}\nPlayed {self.num_plays} animations",
            )

    def tear_down(self) -> None:
        """Run the scene's tear-down hook."""
        self.scene.tear_down()

    def play(
        self,
        *args: Animation | Mobject | _AnimationBuilder,
        subcaption: str | None = None,
        subcaption_duration: float | None = None,
        subcaption_offset: float = 0,
        **kwargs: Any,
    ) -> None:
        """Coordinate an animation request and its optional subcaption."""
        start_time = self.time
        self.renderer.play(self.scene, *args, **kwargs)
        run_time = self.time - start_time

        if subcaption:
            if subcaption_duration is None:
                subcaption_duration = run_time
            # The start of the subcaption needs to be offset by the run time
            # because it is added after the animation has already played.
            self.add_subcaption(
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
        """Create a new output section."""
        self.file_writer.next_section(name, section_type, skip_animations)

    def add_subcaption(
        self, content: str, duration: float = 1, offset: float = 0
    ) -> None:
        """Add a subcaption at the current scene time."""
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
        """Add sound to the scene's output at the current scene time."""
        if self.skip_animations:
            return
        self.file_writer.add_sound(sound_file, self.time + time_offset, gain, **kwargs)
