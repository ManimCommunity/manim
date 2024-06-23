from __future__ import annotations

import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Callable

import numpy as np

from manim import config, logger
from manim.constants import RendererType
from manim.file_writer import FileWriter
from manim.renderer.cairo_renderer import CairoRenderer
from manim.renderer.opengl_renderer import OpenGLRenderer
from manim.renderer.opengl_renderer_window import Window
from manim.scene.scene import Scene, SceneState
from manim.utils.exceptions import EndSceneEarlyException

if TYPE_CHECKING:
    from manim.animation.protocol import AnimationProtocol

    from .camera.camera import Camera
    from .renderer.renderer import RendererProtocol

__all__ = ("Manager",)


class Manager:
    """
    The Brain of Manim

    .. note::

        The only method of this class officially guaranteed to be
        stable is :meth:`~.Manager.render`. Any other methods documented
        are purely for development

    Usage
    -----

        .. code-block:: python

            class Manimation(Scene):
                def construct(self):
                    self.play(FadeIn(Circle()))


            Manager(Manimation).render()
    """

    def __init__(self, scene_cls: type[Scene]) -> None:
        # scene
        self.scene: Scene = scene_cls(self)

        if not isinstance(self.scene, Scene):
            raise ValueError(f"{self.scene!r} is not an instance of Scene")

        self.time = 0

        # Initialize window, if applicable
        if config.preview:
            self.window: Window | None = Window()
        else:
            self.window = None

        # this must be done AFTER instantiating a window
        self.renderer = self.create_renderer()
        self.renderer.use_window()

        # file writer
        self.file_writer = FileWriter(self.scene.get_default_scene_name())  # TODO
        self._write_files = config.write_to_movie

    @property
    def camera(self) -> Camera:
        return self.scene.camera

    def create_renderer(self) -> RendererProtocol:
        match config.renderer:
            case RendererType.OPENGL:
                return OpenGLRenderer()

            case RendererType.CAIRO:
                return CairoRenderer()

            case rendertype:
                raise ValueError(f"Invalid Config Renderer type {rendertype}")

    def _setup(self) -> None:
        """Set up processes and manager"""

        self.scene.setup()

        # these are used for making sure it feels like the correct
        # amount of time has passed in the window instead of rendering
        # at full speed
        self.virtual_animation_start_time = 0
        self.real_animation_start_time = time.perf_counter()

    def render(self) -> None:
        """
        Entry point to running a Manim class

        Example
        -------

        .. code-block:: python

            class MyScene(Scene):
                def construct(self):
                    self.play(Create(Circle()))


            with tempconfig({"preview": True}):
                Manager(MyScene).render()
        """
        self._render_first_pass()
        self._render_second_pass()
        self._interact()

    def _render_first_pass(self) -> None:
        """
        Temporarily use the normal single pass
        rendering system
        """
        self._setup()

        try:
            self.scene.construct()
            self._post_contruct()
            self._interact()
        except EndSceneEarlyException:
            pass
        self._tear_down()

    def _render_second_pass(self) -> None:
        """
        In the future, this method could be used
        for two pass rendering
        """
        ...

    def _post_contruct(self) -> None:
        self.file_writer.finish()
        self._write_files = False

    def _tear_down(self) -> None:
        self.scene.tear_down()

        if config.save_last_frame:
            self._update_frame(0)

        if self.window is not None:
            self.window.close()
            self.window = None

    def _interact(self) -> None:
        if self.window is None:
            return
        logger.info(
            "\nTips: Using the keys `d`, `f`, or `z` "
            + "you can interact with the scene. "
            + "Press `command + q` or `esc` to quit"
        )
        self.scene.skip_animations = False
        self.scene.refresh_static_mobjects()
        while not self.window.is_closing:
            # TODO: Replace with actual dt instead
            # of hardcoded dt
            dt = 1 / config.frame_rate
            self._update_frame(dt)

    def _update_frame(self, dt: float, *, write_to_file: bool | None = None) -> None:
        """Update the current frame by ``dt``

        Parameters
        ----------
            dt : the time in between frames
            write_to_file : Whether to write the result to the output stream.
                Default value checks :attr:`_write_files` to see if it should be written.
        """
        self.time += dt
        self.scene._update_mobjects(dt)

        if self.window is not None:
            self.window.clear()

        state = self.scene.get_state()
        self._render_frame(state, write_file=write_to_file)

        if self.window is not None:
            self.window.swap_buffers()
            # This recursively updates the window with dt=0 until the correct
            # amount of time has passed
            vt = self.time - self.virtual_animation_start_time
            rt = time.perf_counter() - self.real_animation_start_time
            if rt < vt:
                self._update_frame(0, write_to_file=False)

    def _play(
        self, *animations: AnimationProtocol, run_time: float | None = None
    ) -> None:
        """Play a bunch of animations"""
        self.scene.pre_play()

        if self.window is not None:
            self.real_animation_start_time = time.perf_counter()
            self.virtual_animation_start_time = self.time

        self._write_hashed_movie_file()

        self.scene.begin_animations(animations)
        self._progress_through_animations(animations, run_time=run_time)
        self.scene.finish_animations(animations)

        if self.scene.skip_animations and self.window is not None:
            self._update_frame(dt=0)

        self.scene.post_play()

        self.file_writer.end_animation(allow_write=self._write_files)

    def _write_hashed_movie_file(self) -> None:
        """Compute the hash of a self.play call, and write it to a file

        Essentially, a series of methods that need to be called to successfully
        render a frame.
        """
        if not config.write_to_movie:
            return

        if config.disable_caching:
            if not config.disable_caching_warning:
                logger.info("Caching disabled...")
            hash_current_play = f"uncached_{self.file_writer.num_plays:05}"
        else:
            # TODO: Implement some form of caching
            hash_current_play = None

        self.file_writer.add_partial_movie_file(hash_current_play)
        self.file_writer.begin_animation(allow_write=self._write_files)

    def _wait(
        self,
        duration: float,
        *,
        stop_condition: Callable[[], bool] | None = None,
    ) -> None:
        self.scene.pre_play()

        self._write_hashed_movie_file()

        update_mobjects = (
            self.scene.should_update_mobjects()
        )  # TODO: this method needs to be implemented
        condition = stop_condition or (lambda: False)

        last_t = 0
        for t in self._calc_time_progression(duration):
            if update_mobjects:
                dt, last_t = t - last_t, t
                self._update_frame(dt)
                if condition():
                    break
            else:
                self.renderer.render_previous(self.camera)
        self.scene.post_play()

        self.file_writer.end_animation(allow_write=self._write_files)

    def _progress_through_animations(
        self, animations: Iterable[AnimationProtocol], run_time: float | None = None
    ) -> None:
        last_t = 0.0
        run_time = run_time or self._calc_runtime(animations)
        for t in self._calc_time_progression(run_time):
            dt, last_t = t - last_t, t
            self.scene._update_animations(animations, t, dt)
            self._update_frame(dt)

    def _calc_time_progression(self, run_time: float) -> Iterable[float]:
        # we can't use endpoint=True because for two consecutive play calls
        # that would cause an extra frame to be created
        return np.arange(0, run_time, 1 / config.frame_rate)

    def _calc_runtime(self, animations: Iterable[AnimationProtocol]) -> float:
        return max(animation.get_run_time() for animation in animations)

    def _render_frame(self, state: SceneState, write_file: bool | None = None) -> None:
        """Renders a frame based on a state, and writes it to a file"""

        # render the frame to the window
        self.renderer.render(self.scene.camera, state.mobjects)

        should_write = write_file if write_file is not None else self._write_files
        if should_write:
            self._write_frame_from_renderer()

    def _write_frame_from_renderer(self):
        frame = self.renderer.get_pixels()
        self.file_writer.write_frame(frame)
