"""Manager controls playback and advances animation time independently of drawing."""

from unittest.mock import Mock

import pytest

from manim import Animation, Manager, Scene, Square, Wait, tempconfig


@pytest.fixture
def scene():
    # Playback and the clock live in Manager; the trace tests cover both backends.
    with tempconfig(
        {
            "renderer": "cairo",
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


def test_renderer_compatibility_preserves_bootstrap_and_manager_updates(scene):
    renderer = scene.renderer
    assert scene.manager is None
    renderer.time = 2
    manager = Manager(scene)
    assert manager.time == 2
    manager.time = 3
    assert renderer.time == 3
    renderer.time = 4
    assert manager.time == 4
    renderer.num_plays = 4
    assert manager.num_plays == 4
    manager.num_plays = 5
    assert renderer.num_plays == 5
    renderer.skip_animations = True
    assert manager.skip_animations
    manager.skip_animations = False
    assert not renderer.skip_animations


def test_shared_play_compiles_once(scene, monkeypatch):
    compile_data = Mock(wraps=scene.compile_animation_data)
    compile_animations = Mock(wraps=scene.compile_animations)
    monkeypatch.setattr(scene, "compile_animation_data", compile_data)
    monkeypatch.setattr(scene, "compile_animations", compile_animations)
    scene.play(Wait(1, frozen_frame=False))
    assert compile_data.call_count == compile_animations.call_count == 1


def test_raw_backend_drawing_does_not_advance_clock_or_write(scene, monkeypatch):
    scene.add(Square())
    manager = scene._get_manager()
    manager.time = 2
    write = Mock(side_effect=AssertionError("drawing delivered output"))
    monkeypatch.setattr(manager.file_writer, "write_frame", write)
    scene.renderer.render(scene, 0, scene.mobjects)
    assert manager.time == 2
    write.assert_not_called()


def test_cached_play_advances_once_after_its_evaluation_step(scene, monkeypatch):
    """A shortcut is one interval: begin sees its start, finish sees the consumed span."""
    manager = scene._get_manager()
    manager.time = 2
    observed = []

    class Probe(Animation):
        def begin(self):
            observed.append(("begin", scene.time))
            super().begin()

        def interpolate_mobject(self, alpha):
            observed.append(("interpolate", scene.time, float(alpha)))

        def finish(self):
            observed.append(("finish", scene.time))
            super().finish()

    monkeypatch.setattr("manim.manager.get_hash_from_play_call", lambda *a, **k: "hit")
    monkeypatch.setattr(manager.file_writer, "is_already_cached", lambda _: True)
    with tempconfig({"disable_caching": False}):
        scene.play(Probe(Square(), run_time=0.3))
    # 0.3 s at 4 fps is two frames, so the play consumes 0.5 s, as a render would.
    assert observed == [
        ("begin", 2),
        ("interpolate", 2, 0.0),
        ("interpolate", 2, 1.0),
        ("finish", 2.5),
        ("interpolate", 2.5, 1.0),
    ]
    assert manager.time == 2.5


def test_changed_rate_is_rejected_before_execution(scene):
    with tempconfig({"frame_rate": 8}):
        with pytest.raises(ValueError, match="frame_rate changed"):
            scene.wait(1, frozen_frame=False)
        with pytest.raises(ValueError, match="frame_rate changed"):
            scene.render()
    assert scene.time == 0
    assert scene.manager._file_writer is None


def test_skipped_stop_wait_is_stepped_like_a_rendered_one(scene):
    """A skipped stop condition must observe per-frame times, not jump to the end."""
    manager = scene._get_manager()
    manager.time = 2
    scene.renderer._original_skipping_status = True
    stops = []

    def stop():
        stops.append(scene.time)
        return scene.time >= 3

    scene.wait(1, stop_condition=stop)
    assert stops == [2.25, 2.5, 2.75, 3]
    assert scene.time == 3
