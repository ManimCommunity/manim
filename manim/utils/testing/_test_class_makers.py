from __future__ import annotations

from typing import Callable

from manim.file_writer.protocols import FileWriterProtocol
from manim.scene.scene import Scene

from ._frames_testers import _FramesTester


def _make_test_scene_class(
    base_scene: type[Scene],
    construct_test: Callable[[Scene], object],
) -> type[Scene]:
    class _TestedScene(base_scene):
        def construct(self):
            from manim import config

            construct_test(self)

            # Manim hack to render the very last frame (normally the last frame is not the very end of the animation)
            self.wait(1 / config.frame_rate)

    return _TestedScene


class DummySceneFileWriter(FileWriterProtocol):
    """Delegate of SceneFileWriter used to test the frames."""

    def __init__(self, scene_name: str):
        self.num_plays = 0
        self.frames = []

    def begin_animation(self, allow_write: bool = False):
        pass

    def end_animation(self, allow_write: bool = False):
        self.num_plays += 1

    def is_already_cached(self, hash_invocation: str) -> bool:
        return False

    def add_partial_movie_file(self, hash_animation: str) -> None:
        pass

    def write_frame(self, frame):
        self.frames.append(frame)

    def finish(self):
        pass


def _make_scene_file_writer_class(tester: _FramesTester) -> type[FileWriterProtocol]:
    class TestSceneFileWriter(DummySceneFileWriter):
        def write_frame(self, frame):
            tester.check_frame(len(self.frames), frame)
            super().write_frame(frame)

    return TestSceneFileWriter
