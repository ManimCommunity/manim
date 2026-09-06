"""Execution state and orchestration have one owner, independent of pixel delivery."""

from unittest.mock import Mock

import pytest

from manim import Manager, Scene, Square, Wait, tempconfig


@pytest.fixture(params=["cairo", "opengl"])
def scene(request):
    with tempconfig(
        {
            "renderer": request.param,
            "format": "none",
            "live_preview": False,
            "pixel_width": 32,
            "pixel_height": 16,
            "frame_rate": 4,
            "disable_caching": True,
            "progress_bar": "none",
        }
    ):
        scene = Scene()
        try:
            yield scene
        finally:
            scene._get_manager().close()


def test_renderer_bootstrap_is_transferred_not_mirrored(scene):
    renderer = scene.renderer
    assert scene.manager is None
    initial = renderer._pending_execution
    renderer.time = 2
    manager = Manager(scene)
    assert manager._execution is initial
    assert renderer._pending_execution is None
    assert manager.time == 2
    manager.time = 3
    assert renderer.time == 3
    renderer.num_plays = 4
    assert manager.num_plays == 4
    renderer.skip_animations = True
    assert manager.skip_animations
    assert "time" not in renderer.__dict__
    assert "num_plays" not in renderer.__dict__


def test_manager_does_not_delegate_play_or_selection_to_backend(scene, monkeypatch):
    rejected = Mock(side_effect=AssertionError("backend executed policy"))
    monkeypatch.setattr(scene.renderer, "play", rejected)
    monkeypatch.setattr(scene.renderer, "update_skipping_status", rejected)
    scene.play(Wait(1, frozen_frame=False))
    assert scene.manager.num_plays == 1
    rejected.assert_not_called()


def test_clock_advances_when_drawing_and_delivery_are_stubbed(scene, monkeypatch):
    manager = scene._get_manager()
    draw = Mock(return_value=None)
    deliver = Mock()
    monkeypatch.setattr(manager, "_draw_animation_frame", draw)
    monkeypatch.setattr(manager, "_deliver_animation_frame", deliver)
    scene.play(Wait(1, frozen_frame=False))
    assert manager.time == 1
    assert manager.num_plays == 1
    assert draw.call_count == 4
    assert deliver.call_count == 4


def test_raw_backend_drawing_does_not_advance_clock_or_write(scene, monkeypatch):
    scene.add(Square())
    manager = scene._get_manager()
    manager.time = 2
    write = Mock(side_effect=AssertionError("drawing delivered output"))
    monkeypatch.setattr(manager.file_writer, "write_frame", write)
    scene.renderer.render(scene, 0, scene.mobjects)
    assert manager.time == 2
    write.assert_not_called()
