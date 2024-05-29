from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from manim import config, logger
from manim.constants import RendererType

from ..scene.scene import Scene, SceneState
from .opengl_file_writer import FileWriter
from .opengl_renderer import OpenGLRenderer
from .opengl_renderer_window import Window

if TYPE_CHECKING:
    from manim.animation.protocol import AnimationProtocol

    from ..camera.camera import Camera
    from .renderer import RendererProtocol

__all__ = ("Manager",)


class EndSceneError(Exception):
    pass


class Manager:
    """
    The Brain of Manim
    """

    def __init__(self, scene_cls: type[Scene]) -> None:
        # renderer
        self.renderer = self._renderer_class()

        # scene
        self.scene: Scene = scene_cls(self)

        if not isinstance(self.scene, Scene):
            raise ValueError(f"{self.scene!r} is not an instance of Scene")

        self.time = 0

        # Initialize window, if applicable
        if config.preview:
            self.window = Window()
        else:
            self.window = None

        # file writer
        self.file_writer = FileWriter(self.scene.get_default_scene_name())  # TODO
        self.renderer.use_window()

    @property
    def camera(self) -> Camera:
        return self.scene.camera

    @property
    def _renderer_class(self) -> type[RendererProtocol]:
        # TODO: Implement based on renderer in config
        match config.renderer:
            case RendererType.OPENGL:
                return OpenGLRenderer

            case rendertype:
                raise ValueError(f"Invalid Config Renderer type {rendertype}")

    def _setup(self) -> None:
        """Set up processes and manager"""
        if self.file_writer.has_progress_display():
            self.scene.show_animation_progress = False

        self.scene.setup()

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
            self._interact()
        except EndSceneError:
            pass
        except KeyboardInterrupt:
            # Get rid keyboard interrupt symbols
            print("", end="\r")
            self.file_writer.ended_with_interrupt = True
        self._tear_down()

    def _render_second_pass(self) -> None:
        """
        In the future, this method could be used
        for two pass rendering
        """

    def _tear_down(self):
        self.scene.tear_down()

        if config.save_last_frame:
            self._update_frame(0)

        self.file_writer.finish()

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
            dt = 1 / self.camera.fps
            self._update_frame(dt)

    def _update_frame(self, dt: float):
        self.time += dt
        self.scene._update_mobjects(dt)

        if self.window is not None and self.window.is_closing:
            raise EndSceneError()

        if self.window is not None:
            self.window.clear()

        state = self.scene.get_state()
        self._render_frame(state)

        if self.window is not None:
            self.window.swap_buffers()
            vt = self.time - self.virtual_animation_start_time
            rt = time.time() - self.real_animation_start_time
            if rt < vt:
                self._update_frame(0)

    def _play(self, *animations: AnimationProtocol):
        self.scene.pre_play()

        if self.window is not None:
            self.real_animation_start_time = time.time()
            self.virtual_animation_start_time = self.time

        self.scene.begin_animations(animations)
        self._progress_through_animations(animations)
        self.scene.finish_animations(animations)

        if self.scene.skip_animations and self.window is not None:
            self._update_frame(dt=0)

        self.scene.post_play()

    def _progress_through_animations(self, animations: Iterable[AnimationProtocol]):
        last_t = 0
        run_time = self._calc_runtime(animations)
        for t in self._calc_time_progression(run_time):
            dt, last_t = t - last_t, t
            self.scene._update_animations(animations, t, dt)
            self._update_frame(dt)

    def _calc_time_progression(self, run_time: float) -> Iterable[float]:
        return np.arange(0, run_time, 1 / self.camera.fps)

    def _calc_runtime(self, animations: Iterable[AnimationProtocol]):
        return max(animation.get_run_time() for animation in animations)

    def _render_frame(self, state: SceneState) -> Any | None:
        """Renders a frame based on a state, and writes it to a file"""
        data = self._send_scene_to_renderer(state)
        # result = self.file_writer.write(data)

    def _send_scene_to_renderer(self, state: SceneState):
        """Renders the State"""
        result = self.renderer.render(self.camera, state.mobjects)
        return result
