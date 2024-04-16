from __future__ import annotations

import multiprocessing as mp
import queue  # NOTE: Cannot use mp.Queue because of auth keys
import numpy as np
from typing import TYPE_CHECKING, Any, Iterable

from manim import config, logger
from .opengl_renderer import OpenGLRenderer
from .opengl_file_writer import FileWriter

if TYPE_CHECKING:
    from ..scene.scene import SceneState
    from ..camera.camera import Camera

__all__ = ("RenderManager",)


class RenderManager:
    """
    Manage rendering in parallel
    """

    def __init__(self, scene_name: str, camera: Camera, **kwargs) -> None:
        # renderer
        self.renderer = OpenGLRenderer(**kwargs)
        self.ctx = mp.get_context('spawn')

        # file writer
        self.camera = camera
        self.file_writer = FileWriter(scene_name)  # TODO

    def begin(self) -> None:
        """Set up processes and manager"""
        self.processes: queue.Queue[mp.Process] = queue.Queue()
        self.manager = mp.Manager()
        self.manager_dict = self.manager.dict()

    def get_time_progression(self, run_time: float) -> Iterable[float]:
        return np.arange(0, run_time, 1 / self.camera.fps)
        
    def render_state(self, state: SceneState, parallel: bool = True) -> None:
        """Launch a process (optionally in parallel)
        to render a frame
        """
        if parallel and config.parallel:
            logger.warning("Not supported yet")
        self.render_frame(state)

    # type state: SceneState
    def render_frame(self, state: SceneState) -> Any | None:
        """Renders a frame based on a state"""
        data = self.send_scene_to_renderer(state)
        # result = self.file_writer.write(data)
        self.manager_dict[state.time] = data

    def send_scene_to_renderer(self, state: SceneState):
        """Renders the State"""
        result = self.renderer.render(state)
        return result

    def get_frames(self) -> list:
        """Get a list of every frame produced by the
        manager.

        .. warning::
            
            This list is _not guarenteed_ to be sorted until
            after calling :meth:`.RenderManager.finish`
        """
        return self.manager_dict

    def finish(self) -> None:
        for process in self.processes.queue:
            process.join()
        self.manager_dict = dict(sorted(self.manager_dict.items()))
