from __future__ import annotations

import contextlib
import platform
import time
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Callable

import numpy as np
from tqdm import tqdm

from manim import config, logger
from manim.event_handler.window import WindowABC
from manim.file_writer import FileWriter
from manim.plugins import plugins, Hooks
from manim.scene.scene import Scene, SceneState
from manim.utils.exceptions import EndSceneEarlyException

if TYPE_CHECKING:
    import numpy.typing as npt

    from manim.animation.protocol import AnimationProtocol

    from .renderer.renderer import RendererProtocol

__all__ = ("Manager",)


class Manager:
    """
    The Brain of Manim

    .. admonition:: Warning for Developers

        Only methods of this class that are not prefixed with an
        underscore (``_``) are stable. If you override any of the
        ``_`` methods, consider pinning your version of Manim.

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
        self.window = self.create_window()

        # this must be done AFTER instantiating a window
        self.renderer = self.create_renderer()
        self.renderer.use_window()

        # file writer
        self.file_writer = FileWriter(self.scene.get_default_scene_name())
        self._write_files = config.write_to_movie

    # keep these as instance methods so subclasses
    # have access to everything
    def create_renderer(self) -> RendererProtocol:
        """Create and return a renderer instance.

        This can be overridden in subclasses (plugins), if more processing
        is needed.

        Returns
        -------
            An instance of a renderer
        """
        return plugins.renderer()

    def create_window(self) -> WindowABC | None:
        """Create and return a window instance.

        This can be overridden in subclasses (plugins), if more
        processing is needed.

        Returns
        -------
            A window if previewing, else None
        """
        return plugins.window() if config.preview else None

    def setup(self) -> None:
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
        config._warn_about_config_options()
        self._render_first_pass()
        self._render_second_pass()

    def _render_first_pass(self) -> None:
        """
        Temporarily use the normal single pass
        rendering system
        """
        self.setup()

        with contextlib.suppress(EndSceneEarlyException):
            self.scene.construct()
            self.post_contruct()
            self._interact()

        self.tear_down()

    def _render_second_pass(self) -> None:
        """
        In the future, this method could be used
        for two pass rendering
        """
        ...

    def post_contruct(self) -> None:
        """Run post-construct hooks, and clean up the file writer."""
        for hook in plugins.hooks[Hooks.POST_CONSTRUCT]:
            hook(self)

        self.file_writer.finish()
        self._write_files = False

    def tear_down(self) -> None:
        """Tear down the scene and the window."""

        self.scene.tear_down()

        if config.save_last_frame:
            self._update_frame(0)

        if self.window is not None:
            self.window.close()
            self.window = None

    def _interact(self) -> None:
        """Live interaction with the Window"""

        if self.window is None:
            return
        logger.info(
            "\nTips: Using the keys `d`, `f`, or `z` "
            "you can interact with the scene. "
            "Press `command + q` or `esc` to quit"
        )
        self.scene.skip_animations = False
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
        self.scene.time = self.time

        if self.window is not None:
            self.window.clear()

        state = self.scene.get_state()
        self._render_frame(state, write_file=write_to_file)

        if self.window is not None:
            self.window.swap_buffers()
            # This recursively updates the window with dt=0 until the correct
            # amount of time has passed
            # TODO: do ^ better with less overhead
            vt = self.time - self.virtual_animation_start_time
            rt = time.perf_counter() - self.real_animation_start_time
            if rt < vt:
                self._update_frame(0, write_to_file=False)

    def _play(
        self, *animations: AnimationProtocol, run_time: float | None = None
    ) -> None:
        """Play a bunch of animations"""

        self.scene.update_skipping_status()

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
        if hash_current_play is not None and hash_current_play.startswith("uncached_"):
            self.file_writer.begin_animation(allow_write=self._write_files)

    def _create_progressbar(
        self, total: float, description: str, **kwargs
    ) -> tqdm | contextlib.nullcontext[NullProgressBar]:
        """Create a progressbar"""

        if not config.write_to_movie or not config.progress_bar:
            return contextlib.nullcontext(NullProgressBar())
        else:
            return tqdm(
                total=total,
                unit="frames",
                desc=description % {"num": self.file_writer.num_plays},
                ascii=True if platform.system() == "Windows" else None,
                leave=config.progress_bar == "leave",
                disable=config.progress_bar == "none",
                **kwargs,
            )

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

        progression = self._calc_time_progression(duration)
        with self._create_progressbar(
            progression.shape[0], "Waiting %(num)d: "
        ) as progress:
            last_t = 0
            for t in progression:
                dt, last_t = t - last_t, t
                if update_mobjects:
                    self._update_frame(dt)
                    if condition():
                        progress.update(duration - t)
                        break
                else:
                    # if we don't need to update mobjects
                    # we can just leave the mobjects on the window
                    # and increment the time
                    # but we still have to write frames
                    self.time += dt
                    self.write_frame()
                progress.update(1)
        self.scene.post_play()

        self.file_writer.end_animation(allow_write=self._write_files)

    def _progress_through_animations(
        self, animations: Sequence[AnimationProtocol], run_time: float | None = None
    ) -> None:
        last_t = 0.0
        run_time = run_time or self._calc_runtime(animations)
        progression = self._calc_time_progression(run_time)
        with self._create_progressbar(
            progression.shape[0],
            f"Animation %(num)d: {animations[0]}{', etc.' if len(animations) > 1 else ''}",
        ) as progress:
            for t in self._calc_time_progression(run_time):
                dt, last_t = t - last_t, t
                self.scene._update_animations(animations, t, dt)
                self._update_frame(dt)
                progress.update(1)

    def _calc_time_progression(self, run_time: float) -> npt.NDArray[np.float64]:
        return np.arange(0, run_time, 1 / config.frame_rate)

    def _calc_runtime(self, animations: Iterable[AnimationProtocol]) -> float:
        return max(animation.get_run_time() for animation in animations)

    def _render_frame(self, state: SceneState, write_file: bool | None = None) -> None:
        """Renders a frame based on a state, and writes it to a file"""

        # render the frame to the window
        self.renderer.render(self.scene.camera, state.mobjects)

        should_write = write_file if write_file is not None else self._write_files
        if should_write:
            self.write_frame()

    def write_frame(self):
        """Take a frame from the renderer and write it in the file writer."""

        frame = self.renderer.get_pixels()
        self.file_writer.write_frame(frame)


class NullProgressBar:
    """Fake progressbar."""

    def update(self, _) -> None: ...
