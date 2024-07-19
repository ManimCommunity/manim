from __future__ import annotations

__all__ = ["Manager"]

import contextlib
import platform
import time
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import numpy as np
from tqdm import tqdm

from manim import config, logger
from manim.event_handler.window import WindowABC
from manim.file_writer import FileWriter
from manim.plugins import Hooks, plugins
from manim.scene.scene import Scene, SceneState
from manim.utils.exceptions import EndSceneEarlyException
from manim.utils.hashing import get_hash_from_play_call

if TYPE_CHECKING:
    import numpy.typing as npt
    from typing_extensions import Any

    from manim.animation.protocol import AnimationProtocol
    from manim.file_writer.protocols import FileWriterProtocol
    from manim.renderer.renderer import RendererProtocol

Scene_co = TypeVar("Scene_co", covariant=True, bound=Scene)


class Manager(Generic[Scene_co]):
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

    def __init__(self, scene_cls: type[Scene_co]) -> None:
        # scene
        self.scene: Scene_co = scene_cls(manager=self)

        if not isinstance(self.scene, Scene):
            raise ValueError(f"{self.scene!r} is not an instance of Scene")

        self.time = 0.0

        # Initialize window, if applicable
        self.window = self.create_window()

        # this must be done AFTER instantiating a window
        self.renderer = self.create_renderer()
        self.renderer.use_window()

        # file writer
        self.file_writer: FileWriterProtocol = self.create_file_writer()
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
        renderer = plugins.renderer()
        if config.preview:
            renderer.use_window()
        return renderer

    def create_window(self) -> WindowABC | None:
        """Create and return a window instance.

        This can be overridden in subclasses (plugins), if more
        processing is needed.

        Returns
        -------
            A window if previewing, else None
        """
        return plugins.window() if config.preview else None

    def create_file_writer(self) -> FileWriterProtocol:
        """Create and returna file writer instance.

        This can be overridden in subclasses (plugins), if more
        processing is needed.

        Returns
        -------
            A file writer satisfying :class:`.FileWriterProtocol`
        """
        return FileWriter(scene_name=self.scene.get_default_scene_name())

    def setup(self) -> None:
        """Set up processes and manager"""

        self.scene.setup()

        # these are used for making sure it feels like the correct
        # amount of time has passed in the window instead of rendering
        # at full speed
        # See the docstring of :meth:`_wait_for_animation_time`
        self.virtual_animation_start_time = 0.0
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

        if self.file_writer.num_plays:
            self.file_writer.finish()
        # otherwise no animations were played
        elif config.write_to_movie or config.save_last_frame:
            self.render_state(write_frame=False)
            # FIXME: for some reason the OpenGLRenderer does not give out the
            # correct frame values here
            frame = self.renderer.get_pixels()
            # NOTE: add hooks for post-processing (e.g. gaussian blur)?
            self.file_writer.save_image(frame)

        self._write_files = False

    def tear_down(self) -> None:
        """Tear down the scene and the window."""

        self.scene.tear_down()

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
        # TODO: Replace with actual dt instead
        # of hardcoded dt
        dt = 1 / config.frame_rate
        while not self.window.is_closing:
            self._update_frame(dt)

    def _update_frame(self, dt: float, *, write_frame: bool | None = None) -> None:
        """Update the current frame by ``dt``

        Parameters
        ----------
            dt : the time in between frames
            write_frame : Whether to write the result to the output stream (videos ONLY).
                Default value checks :attr:`_write_files` to see if it should be written.
        """
        self.time += dt
        self.scene._update_mobjects(dt)
        self.scene.time = self.time

        if self.window is not None:
            self.window.clear()

        self.render_state(write_frame=write_frame)

        if self.window is not None:
            self.window.swap_buffers()
            self._wait_for_animation_time()

    def _wait_for_animation_time(self) -> None:
        """Wait for the real time to catch up to the "virtual" animation time.

        Animations can render faster than real time, so we have to
        slow the window down for the correct amount of time, such
        as during a wait animation.
        """

        if self.window is None:
            return

        vt = self.time - self.virtual_animation_start_time
        rt = time.perf_counter() - self.real_animation_start_time
        # we can't sleep because we still need to poll for events,
        # e.g. hitting Escape or close
        while rt < vt:
            # make sure to poll for events
            self.window.swap_buffers()
            rt = time.perf_counter() - self.real_animation_start_time

    def _play(self, *animations: AnimationProtocol) -> None:
        """Play a bunch of animations"""

        if self.window is not None:
            self.real_animation_start_time = time.perf_counter()
            self.virtual_animation_start_time = self.time

        self._write_hashed_movie_file(animations)

        self.scene.begin_animations(animations)
        self._progress_through_animations(animations)
        self.scene.finish_animations(animations)

        self.scene.post_play()

        self.file_writer.end_animation(allow_write=self._write_files)

    def _write_hashed_movie_file(self, animations: Sequence[AnimationProtocol]) -> None:
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
            hash_current_play = get_hash_from_play_call(
                self.scene,
                self.scene.camera,
                animations,
                self.scene.mobjects,
            )
            if self.file_writer.is_already_cached(hash_current_play):
                logger.info(
                    f"Animation {self.file_writer.num_plays} : Using cached data (hash : {hash_current_play})"
                )
                # TODO: think about how to skip
                raise NotImplementedError(
                    "Skipping cached animations is not implemented yet"
                )

        self.file_writer.add_partial_movie_file(hash_current_play)
        self.file_writer.begin_animation(allow_write=self._write_files)

    def _create_progressbar(
        self, total: float, description: str, **kwargs: Any
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

    # TODO: change to a single wait animation
    def _wait(
        self,
        duration: float,
        *,
        stop_condition: Callable[[], bool] | None = None,
    ) -> None:
        self.scene.pre_play()

        self._write_hashed_movie_file(animations=[])

        update_mobjects = self.scene.should_update_mobjects()
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
        self, animations: Sequence[AnimationProtocol]
    ) -> None:
        last_t = 0.0
        run_time = self._calc_runtime(animations)
        progression = self._calc_time_progression(run_time)
        with self._create_progressbar(
            progression.shape[0],
            f"Animation %(num)d: {animations[0]}{', etc.' if len(animations) > 1 else ''}",
        ) as progress:
            for t in progression:
                dt, last_t = t - last_t, t
                self.scene._update_animations(animations, t, dt)
                self._update_frame(dt)
                progress.update(1)

    def _calc_time_progression(self, run_time: float) -> npt.NDArray[np.float64]:
        """Compute the time values at which to evaluate the animation"""

        return np.arange(0, run_time, 1 / config.frame_rate)

    def _calc_runtime(self, animations: Iterable[AnimationProtocol]) -> float:
        """Calculate the runtime of an iterable of animations.

        .. warning::

            If animations is a generator, this will consume the generator.
        """
        return max(animation.get_run_time() for animation in animations)

    def render_state(self, write_frame: bool | None = None) -> None:
        """Render the current state of the scene.

        Any extra kwargs are passed to :meth:`_render_frame`.
        """
        state = self.scene.get_state()
        self._render_frame(state, write_frame=write_frame)

    def _render_frame(
        self, state: SceneState, *, write_frame: bool | None = None
    ) -> None:
        """Renders a frame based on a state, and writes it to the file writers stream.

        This is used for writing a single frame. Any extra kwargs are passed to :meth:`write_frame`.

        .. warning::

            This method will not work if :meth:`.FileWriter.begin_animation` and
            :meth:`.FileWriter.add_partial_movie_file` have not been called. Do NOT
            use this to write a single frame!
        """

        # TODO: change self.scene.camera to state.camera
        self.renderer.render(self.scene.camera, state.mobjects)

        should_write = write_frame if write_frame is not None else self._write_files
        if should_write:
            self.write_frame()

    def write_frame(self) -> None:
        """Take a frame from the renderer and write it in the file writer."""

        frame = self.renderer.get_pixels()
        self.file_writer.write_frame(frame)


class NullProgressBar:
    """Fake progressbar."""

    def update(self, _: Any) -> None: ...
