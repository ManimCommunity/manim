"""OpenGL setup and snapshots do not create output writers."""

from unittest.mock import Mock

from manim import Scene
from manim.renderer.opengl import OpenGLRenderer
from manim.scene.scene_file_writer import SceneFileWriter


def test_opengl_writer_demand_is_separate_from_context_and_snapshot(
    config, using_temp_opengl_config
):
    config.format = "png"
    config.pixel_width = 64
    config.pixel_height = 32
    config.live_preview = False
    factory = Mock(wraps=SceneFileWriter)
    renderer = OpenGLRenderer(file_writer_class=factory)
    scene = Scene(renderer=renderer)
    try:
        factory.assert_not_called()
        assert scene.manager is None
        assert scene.get_image().size == (64, 32)
        assert scene.manager._file_writer is None
        factory.assert_not_called()
        writer = renderer.file_writer
        assert scene.manager.file_writer is writer
        factory.assert_called_once_with(scene.file_writer_settings)
    finally:
        if scene.manager is not None and scene.manager._file_writer is not None:
            scene.manager._file_writer.abort_encode_jobs()
        renderer.close()
