from __future__ import annotations

import multiprocessing as mp
import queue  # NOTE: Cannot use mp.Queue because of auth keys
import time
from typing import TYPE_CHECKING, Any, Iterable

from manim import config
from .opengl_renderer import OpenGLRenderer
from .opengl_file_writer import FileWriter

if TYPE_CHECKING:
    from ..scene.scene import SceneState

__all__ = ("RenderManager",)


class RenderManager:
    """
    Manage rendering in parallel
    """

    def __init__(self, scene_name: str, **kwargs) -> None:
        # renderer
        self.renderer = OpenGLRenderer(**kwargs)
        self.ctx = mp.get_context('spawn')

        # file writer
        self.file_writer = FileWriter(scene_name)  # TODO

    def begin(self) -> None:
        """Set up processes and manager"""
        self.processes: queue.Queue[mp.Process] = queue.Queue()
        self.manager = mp.Manager()
        self.manager_dict = self.manager.dict()

    def start_dt_calculations(self) -> None:
        self.last_t = time.perf_counter()

    def refresh_dt(self) -> float:
        dt = time.perf_counter() - self.last_t
        self.last_t = time.perf_counter()
        return dt

    def get_time_progression(self, run_time: float) -> Iterable[float]:
        while (dt := self.refresh_dt()) < run_time:
            yield dt

    def render_state(self, state: SceneState, parallel: bool = True) -> None:
        """Launch a process (optionally in parallel)
        to render a frame
        """
        if parallel and config.in_parallel:
            process = mp.Process(
                target=self.render_frame,
                args=(state,),
                name=str(state.time)
            )
            self.processes.put(process)
            process.start()
        else:
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
        for process in iter(self.processes.get, None):
            process.join()
        self.manager_dict = dict(sorted(self.manager_dict.items()))
