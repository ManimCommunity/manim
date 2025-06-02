from __future__ import annotations

from typing import Any, Callable

from manim.renderer.cairo_renderer import CairoRenderer
from manim.renderer.opengl_renderer import OpenGLRenderer
from manim.scene.scene import Scene
from manim.scene.scene_file_writer import SceneFileWriter
from manim.typing import PixelArray, StrPath

from ._frames_testers import _FramesTester


def _make_test_scene_class(
    base_scene: type[Scene],
    construct_test: Callable[[Scene], None],
    test_renderer: CairoRenderer | OpenGLRenderer | None,
) -> type[Scene]:
    # TODO: Get the type annotation right for the base_scene argument.
    class _TestedScene(base_scene):  # type: ignore[valid-type, misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, renderer=test_renderer, **kwargs)

        def construct(self) -> None:
            construct_test(self)

            # Manim hack to render the very last frame (normally the last frame is not the very end of the animation)
            if self.animations is not None:
                self.update_to_time(self.get_run_time(self.animations))
                self.renderer.render(self, 1, self.moving_mobjects)

    return _TestedScene


def _make_test_renderer_class(from_renderer: type) -> Any:
    # Just for inheritance.
    class _TestRenderer(from_renderer):
        pass

    return _TestRenderer


class DummySceneFileWriter(SceneFileWriter):
    """Delegate of SceneFileWriter used to test the frames."""

    def __init__(
        self,
        renderer: CairoRenderer | OpenGLRenderer,
        scene_name: StrPath,
        **kwargs: Any,
    ) -> None:
        super().__init__(renderer, scene_name, **kwargs)
        self.i = 0

    def init_output_directories(self, scene_name: StrPath) -> None:
        pass

    def add_partial_movie_file(self, hash_animation: str) -> None:
        pass

    def begin_animation(
        self, allow_write: bool = True, file_path: StrPath | None = None
    ) -> Any:
        pass

    def end_animation(self, allow_write: bool = False) -> None:
        pass

    def combine_to_movie(self) -> None:
        pass

    def combine_to_section_videos(self) -> None:
        pass

    def clean_cache(self) -> None:
        pass

    def write_frame(
        self, frame_or_renderer: PixelArray | OpenGLRenderer, num_frames: int = 1
    ) -> None:
        self.i += 1


def _make_scene_file_writer_class(tester: _FramesTester) -> type[SceneFileWriter]:
    class TestSceneFileWriter(DummySceneFileWriter):
        def write_frame(
            self, frame_or_renderer: PixelArray | OpenGLRenderer, num_frames: int = 1
        ) -> None:
            tester.check_frame(self.i, frame_or_renderer)
            super().write_frame(frame_or_renderer, num_frames=num_frames)

    return TestSceneFileWriter
