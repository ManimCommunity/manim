"""Check scene time during timeline capture and the placement of offset calls."""

import wave
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from manim import Dot, Manager, Scene, tempconfig


def test_time_going_backwards_fails_the_capture():
    class Rewinds(Scene):
        def construct(self):
            self.wait(1, frozen_frame=False)
            self.renderer.time = 0.5
            self.wait(1, frozen_frame=False)

    with tempconfig({"format": "none", "frame_rate": 4, "progress_bar": "none"}):
        manager = Manager(Rewinds())
        with pytest.raises(ValueError, match="goes backwards"):
            manager.evaluate(capture_timeline=True)
        with pytest.raises(RuntimeError, match="No completed"):
            manager.timeline


def test_offsets_move_sounds_and_captions_but_not_the_clock():
    class Offsets(Scene):
        def construct(self):
            self.wait(1, frozen_frame=False)
            self.add_sound("past.wav", time_offset=-0.5)
            self.add_sound("future.wav", time_offset=2)
            self.add_subcaption("past", offset=-1)
            self.wait(1, frozen_frame=False)

    with tempconfig({"format": "none", "frame_rate": 4, "progress_bar": "none"}):
        manager = Manager(Offsets())
        manager.evaluate(capture_timeline=True)
    data = manager.timeline.to_dict()
    assert data["duration"] == 2
    calls = [entry for entry in data["entries"] if entry["kind"] != "wait"]
    assert [entry["start"] for entry in calls] == [0.5, 3, 0]


def test_values_json_cannot_hold_are_recorded_in_a_readable_form():
    class OddValues(Scene):
        def construct(self):
            self.next_section("intro")
            self.add_sound(Path("click.wav"), gain=np.float32(-3), seed=np.int64(2))
            self.add_subcaption("caption", duration=np.float64(0.5))
            self.wait(0.5, frozen_frame=False)

    with tempconfig({"format": "none", "frame_rate": 4, "progress_bar": "none"}):
        manager = Manager(OddValues())
        manager.evaluate(capture_timeline=True)
    entries = manager.timeline.to_dict()["entries"]
    sound = next(entry for entry in entries if entry["kind"] == "sound")
    caption = next(entry for entry in entries if entry["kind"] == "caption")
    assert (sound["file"], sound["gain"], sound["options"]) == (
        "click.wav",
        -3.0,
        {"seed": 2},
    )
    assert caption["end"] - caption["start"] == 0.5


@pytest.mark.parametrize("mode", ["evaluate", "render"])
def test_calls_before_the_run_are_recorded_in_both_modes(tmp_path, mode):
    class Introduced(Scene):
        def __init__(self):
            super().__init__()
            self.next_section("intro")

        def construct(self):
            self.wait(0.25)

    with tempconfig(
        {
            "format": "none",
            "frame_rate": 4,
            "progress_bar": "none",
            "media_dir": str(tmp_path),
        }
    ):
        scene = Introduced()
        getattr(scene.manager, mode)(capture_timeline=True)
    entries = scene.manager.timeline.to_dict()["entries"]
    assert [entry["name"] for entry in entries if entry["kind"] == "section"] == [
        "intro"
    ]


def test_a_failed_play_leaves_no_entries_of_calls_made_during_it(tmp_path):
    click = tmp_path / "click.wav"
    with wave.open(str(click), "wb") as sound:
        sound.setnchannels(1)
        sound.setsampwidth(2)
        sound.setframerate(8000)
        sound.writeframes(bytes(1600))

    class Retries(Scene):
        def construct(self):
            def sound_then_fail(mobject, dt):
                if self.time >= 0.25:
                    self.add_sound(str(click))
                    raise ValueError("updater failed")

            dot = Dot().add_updater(sound_then_fail)
            self.add(dot)
            with pytest.raises(ValueError, match="updater failed"):
                self.wait(0.5)
            dot.clear_updaters()
            self.wait(0.5)

    with tempconfig(
        {
            "format": "mp4",
            "frame_rate": 4,
            "progress_bar": "none",
            "disable_caching": True,
            "media_dir": str(tmp_path),
        }
    ):
        manager = Manager(Retries())
        manager.render(capture_timeline=True)
    kinds = [entry["kind"] for entry in manager.timeline.to_dict()["entries"]]
    assert kinds == ["wait"]


def test_a_run_without_a_timeline_does_not_look_up_the_scene_file(monkeypatch):
    lookup = Mock(side_effect=AssertionError("looked up"))
    monkeypatch.setattr("manim._run.record._source_file", lookup)
    with tempconfig({"format": "none", "frame_rate": 4, "progress_bar": "none"}):
        manager = Manager(Scene())
        manager.evaluate()
    lookup.assert_not_called()
