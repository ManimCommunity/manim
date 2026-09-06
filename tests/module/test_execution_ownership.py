"""Execution state and orchestration have one owner, independent of pixel delivery."""

from unittest.mock import Mock

import pytest

from manim import Animation, Manager, Scene, Square, Wait, tempconfig


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
    draw_times = []
    delivery_times = []
    draw = Mock(side_effect=lambda _: draw_times.append(manager.time))
    deliver = Mock(side_effect=lambda *_: delivery_times.append(manager.time))
    monkeypatch.setattr(manager, "_draw_animation_frame", draw)
    monkeypatch.setattr(manager, "_deliver_animation_frame", deliver)
    scene.play(Wait(1, frozen_frame=False))
    assert manager.time == 1
    assert manager.num_plays == 1
    assert draw.call_count == 4
    assert deliver.call_count == 4
    assert draw_times == [0, 0.25, 0.5, 0.75]
    assert delivery_times == [0.25, 0.5, 0.75, 1]


def test_raw_backend_drawing_does_not_advance_clock_or_write(scene, monkeypatch):
    scene.add(Square())
    manager = scene._get_manager()
    manager.time = 2
    write = Mock(side_effect=AssertionError("drawing delivered output"))
    monkeypatch.setattr(manager.file_writer, "write_frame", write)
    scene.renderer.render(scene, 0, scene.mobjects)
    assert manager.time == 2
    write.assert_not_called()


def test_stopped_clock_and_next_play_without_frame_delivery(scene, monkeypatch):
    manager = scene._get_manager()
    manager.time = 2
    monkeypatch.setattr(manager, "_draw_animation_frame", Mock(return_value=None))
    monkeypatch.setattr(manager, "_deliver_animation_frame", Mock())
    scene.wait(1, stop_condition=lambda: scene.time >= 2.5)
    assert manager.time == 2.5
    scene.wait(0.3, frozen_frame=False)
    assert manager.time == 3
    assert manager.num_plays == 2


def test_cached_play_advances_once_before_begin(scene, monkeypatch):
    manager = scene._get_manager()
    manager.time = 2
    observed = []

    class Probe(Animation):
        def begin(self):
            observed.append(("begin", scene.time))
            super().begin()

        def finish(self):
            observed.append(("finish", scene.time))
            super().finish()

    monkeypatch.setattr("manim.manager.get_hash_from_play_call", lambda *a, **k: "hit")
    monkeypatch.setattr(manager.file_writer, "is_already_cached", lambda _: True)
    with tempconfig({"disable_caching": False}):
        scene.play(Probe(Square(), run_time=0.3))
    assert observed == [("begin", 2.3), ("finish", 2.3)]
    assert manager.time == 2.3


def test_frozen_clock_does_not_depend_on_output_delivery(scene, monkeypatch):
    manager = scene._get_manager()
    write = Mock()
    monkeypatch.setattr(manager.file_writer, "write_frame", write)
    with tempconfig({"format": "none"}):
        scene.wait(0.3, frozen_frame=True)
        scene.wait(0.3, frozen_frame=True)
    assert manager.time == 0.5


def test_sample_clock_uses_progression_rate_not_raster_rate(scene):
    # Progression already resolves the current config at play time. A stale
    # Cairo raster rate must not make a 1s/8-sample play consume 2 semantic seconds.
    with tempconfig({"frame_rate": 8}):
        scene.wait(1, frozen_frame=False)
    assert scene.time == 1


def test_skipped_stop_wait_keeps_legacy_nominal_span(scene):
    manager = scene._get_manager()
    manager.time = 2
    scene.renderer._original_skipping_status = True
    stops = []

    def stop():
        stops.append(scene.time)
        return scene.time >= 3

    scene.wait(1, stop_condition=stop)
    assert stops == [3]
    assert scene.time == 3
