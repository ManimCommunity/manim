"""Deferred OpenGL opening, native retirement, and cold inspection."""

from concurrent.futures import ThreadPoolExecutor
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


@pytest.mark.parametrize("live_preview", [False, True])
def test_scene_construction_and_cold_close_open_nothing(
    scene_factory, monkeypatch, live_preview
):
    from manim.renderer.opengl import window as window_module

    create_context = Mock(side_effect=AssertionError("unexpected context"))
    create_window = Mock(side_effect=AssertionError("unexpected window"))
    monkeypatch.setattr(moderngl, "create_context", create_context)
    monkeypatch.setattr(window_module, "Window", create_window)
    with tempconfig({"live_preview": live_preview}):
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
    renderer.close()
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
    window = Mock(side_effect=AssertionError("inspection opened a preview"))
    monkeypatch.setattr(window_module, "Window", window)
    pixels = np.asarray(second.get_image())
    assert pixels.shape == (32, 64, 4)
    assert second.renderer._context is None
    assert second.renderer.window is None
    assert second.manager._file_writer is None
    window.assert_not_called()
    # Read directly, without renderer access that could hide a missing restore.
    assert target.read(components=4) == before


def test_closed_ordinary_scene_can_use_new_snapshot_scope(scene_factory):
    scene = scene_factory()
    scene.add(Square(fill_opacity=1))
    before = np.asarray(scene.get_image())
    scene.renderer.close()
    np.testing.assert_array_equal(scene.get_image(), before)
    assert scene.renderer._context is None
    assert scene.renderer._closed


def test_thread_affinity_starts_at_resource_open_not_scene_construction(scene_factory):
    with ThreadPoolExecutor(1) as pool:
        scene = pool.submit(scene_factory).result()
    scene.renderer.open()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(scene.renderer.close)
        with pytest.raises(RuntimeError, match="owning thread"):
            future.result()
    assert not scene.renderer._closed
    scene.renderer.close()


def test_failed_host_retirement_can_be_retried(scene_factory, monkeypatch):
    create_context = moderngl.create_context
    failure = KeyboardInterrupt("context release interrupted")
    calls = []

    def create(*args, **kwargs):
        context = create_context(*args, **kwargs)
        original = context.release

        def release():
            calls.append(context)
            if len(calls) == 1:
                raise failure
            original()

        monkeypatch.setattr(context, "release", release)
        return context

    monkeypatch.setattr(moderngl, "create_context", create)
    scene = scene_factory()
    renderer = scene.renderer
    renderer.open()
    context = renderer.context
    with pytest.raises(KeyboardInterrupt) as caught:
        renderer.close()
    assert caught.value is failure
    assert renderer._context is context
    assert not renderer._closed
    with pytest.raises(RuntimeError, match="retiring"):
        renderer.get_frame()
    renderer.close()
    assert renderer._closed
    assert isinstance(context.mglo, moderngl.InvalidObject)


def test_rebinding_uses_new_resources_without_changing_legacy_clock(scene_factory):
    first = scene_factory()
    renderer = first.renderer
    renderer.open()
    old_context = renderer.context
    renderer.time = 2.5
    second = Scene(renderer=renderer)
    assert isinstance(old_context.mglo, moderngl.InvalidObject)
    assert renderer._context is None
    assert renderer.time == 2.5
    renderer.update_frame(second)
    assert renderer.context is not old_context
