"""OpenGL resources open on demand, close after use, and support fresh snapshots."""

from unittest.mock import Mock

import moderngl
import numpy as np
import pytest
from PIL import Image

from manim import BLUE, RED, Scene, Square, tempconfig
from manim.renderer.opengl.shader import shader_program_cache


@pytest.fixture
def scene_factory(using_temp_opengl_config):
    scenes = []
    with tempconfig(
        {"format": "none", "live_preview": False, "pixel_width": 64, "pixel_height": 32}
    ):

        def create():
            scene = Scene()
            scenes.append(scene)
            return scene

        try:
            yield create
        finally:
            for scene in reversed(scenes):
                scene.renderer.close()


def test_scene_construction_and_cold_close_open_nothing(scene_factory, monkeypatch):
    from manim.renderer.opengl import window as window_module

    create_context = Mock(side_effect=AssertionError("unexpected context"))
    create_window = Mock(side_effect=AssertionError("unexpected window"))
    monkeypatch.setattr(moderngl, "create_context", create_context)
    monkeypatch.setattr(window_module, "Window", create_window)
    with tempconfig({"live_preview": True}):
        scene = scene_factory()
    assert scene.renderer._context is None
    assert scene.renderer.window is None
    assert scene.manager is None
    scene.renderer.close()
    scene.renderer.close()
    with pytest.raises(RuntimeError, match="closed"):
        scene.renderer.get_frame()
    create_context.assert_not_called()
    create_window.assert_not_called()


def test_render_opens_before_setup_and_close_retires_real_objects(
    scene_factory, monkeypatch, tmp_path
):
    scene = scene_factory()
    renderer = scene.renderer
    scene.add(Square(fill_opacity=1))

    def setup():
        assert renderer._context is not None
        assert scene.manager._file_writer is not None

    monkeypatch.setattr(scene, "setup", setup)
    with scene._get_manager():
        scene.render()
        image_path = tmp_path / "texture.png"
        Image.new("RGBA", (4, 4), "red").save(image_path)
        renderer.get_texture_id(str(image_path))
        context = renderer.context
        frame = renderer.frame_buffer_object
        program = context.program(
            vertex_shader="#version 330\nvoid main(){gl_Position=vec4(0);}",
            fragment_shader="#version 330\nout vec4 color;void main(){color=vec4(1);}",
        )
        monkeypatch.setitem(shader_program_cache, "lifecycle-owned", program)
        unrelated = Mock(ctx=object())
        monkeypatch.setitem(shader_program_cache, "lifecycle-unrelated", unrelated)
        resources = [
            context,
            frame,
            *frame.color_attachments,
            frame.depth_attachment,
            program,
            *renderer._textures,
        ]
    assert all(
        isinstance(resource.mglo, moderngl.InvalidObject) for resource in resources
    )
    assert "lifecycle-owned" not in shader_program_cache
    unrelated.release.assert_not_called()
    assert renderer._context is None
    assert renderer._textures == []
    assert renderer.path_to_texture_id == {}
    with pytest.raises(RuntimeError, match="closed"):
        renderer.open()


def test_cold_snapshot_never_opens_preview_and_restores_previous_host(
    scene_factory, monkeypatch
):
    from manim.renderer.opengl import window as window_module

    first = scene_factory()
    first.add(Square(fill_color=RED, fill_opacity=1))
    first.renderer.update_frame(first)
    target = first.renderer.frame_buffer_object
    before = target.read(components=4)
    with tempconfig({"live_preview": True}):
        second = scene_factory()
    second.add(Square(fill_color=BLUE, fill_opacity=1))
    second.renderer.close()
    window = Mock(side_effect=AssertionError("inspection opened a preview"))
    monkeypatch.setattr(window_module, "Window", window)
    pixels = np.asarray(second.get_image())
    assert pixels.shape == (32, 64, 4)
    assert second.renderer._context is None
    assert second.renderer._closed
    assert second.renderer.window is None
    assert second.manager._file_writer is None
    window.assert_not_called()
    # Read directly, without renderer access that could hide a missing restore.
    assert target.read(components=4) == before
