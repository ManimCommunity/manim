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
    # Even an old subclass method with this name is no longer the execution seam.
    monkeypatch.setattr(scene, "play_internal", rejected, raising=False)
    scene.play(Wait(1, frozen_frame=False))
    assert scene.manager.num_plays == 1
    rejected.assert_not_called()


def test_late_manager_cannot_claim_a_rebound_renderers_state(scene):
    old = Scene()
    current = Scene(renderer=old.renderer)
    try:
        with pytest.raises(RuntimeError, match="renderer was rebound"):
            Manager(old)
        assert old.manager is None
        with current._get_manager() as manager:
            assert manager.time == 0
    finally:
        current.renderer.close()


def test_closed_backend_rejected_before_output_startup(scene):
    manager = scene._get_manager()
    scene.renderer.close()
    with pytest.raises(RuntimeError, match="closed"):
        scene.play(Wait(1, frozen_frame=False))
    assert manager._file_writer is None


def test_shared_play_compiles_once(scene, monkeypatch):
    compile_data = Mock(wraps=scene.compile_animation_data)
    compile_animations = Mock(wraps=scene.compile_animations)
    monkeypatch.setattr(scene, "compile_animation_data", compile_data)
    monkeypatch.setattr(scene, "compile_animations", compile_animations)
    scene.play(Wait(1, frozen_frame=False))
    assert compile_data.call_count == compile_animations.call_count == 1


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
    assert observed == [("begin", 2.5), ("finish", 2.5)]
    assert manager.time == 2.5


def test_frozen_clock_does_not_depend_on_output_delivery(scene, monkeypatch):
    manager = scene._get_manager()
    write = Mock()
    monkeypatch.setattr(manager.file_writer, "write_frame", write)
    with tempconfig({"format": "none"}):
        scene.wait(0.3, frozen_frame=True)
        scene.wait(0.3, frozen_frame=True)
    assert manager.time == 0.5


def test_changed_rate_is_rejected_before_execution(scene):
    with tempconfig({"frame_rate": 8}):
        with pytest.raises(ValueError, match="frame_rate changed"):
            scene.wait(1, frozen_frame=False)
        with pytest.raises(ValueError, match="frame_rate changed"):
            scene.render()
    assert scene.time == 0
    assert scene.manager._file_writer is None


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
