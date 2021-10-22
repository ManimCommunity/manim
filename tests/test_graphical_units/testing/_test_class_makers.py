from typing import Callable, Type

from manim.scene.scene import Scene
from manim.scene.scene_file_writer import SceneFileWriter

from ._frames_testers import _FramesTester


def _make_test_scene_class(
    base_scene: Type[Scene],
    construct_test: Callable[[Scene], None],
    test_renderer,
) -> Type[Scene]:
    class _TestedScene(base_scene):
        def __init__(self, *args, **kwargs):
            super().__init__(renderer=test_renderer, *args, **kwargs)

        def construct(self):
            construct_test(self)

            # Manim hack to render the very last frame (normally the last frame is not the very end of the animation)
            if self.animations is not None:
                self.update_to_time(self.get_run_time(self.animations))
                self.renderer.render(self, 1, self.moving_mobjects)

    return _TestedScene


def _make_test_renderer_class(from_renderer):
    # Just for inheritance.
    class _TestRenderer(from_renderer):
        pass

    return _TestRenderer


class DummySceneFileWriter(SceneFileWriter):
    """Delegate of SceneFileWriter used to test the frames."""

    def __init__(self, renderer, scene_name, **kwargs):
        super().__init__(renderer, scene_name, **kwargs)
        self.i = 0

    def init_output_directories(self, scene_name):
        pass

    def next_section(self, type, name=None):
        pass

    def add_partial_movie_file(self, hash_animation):
        pass

    def begin_animation(self, allow_write=True):
        pass

    def end_animation(self, allow_write):
        pass

    def combine_to_movie(self):
        pass

    def combine_to_section_videos(self):
        pass

    def clean_cache(self):
        pass

    def write_frame(self, frame_or_renderer):
        self.i += 1


def _make_scene_file_writer_class(tester: _FramesTester) -> Type[SceneFileWriter]:
    class TestSceneFileWriter(DummySceneFileWriter):
        def write_frame(self, frame_or_renderer):
            tester.check_frame(self.i, frame_or_renderer)
            super().write_frame(frame_or_renderer)

    return TestSceneFileWriter
