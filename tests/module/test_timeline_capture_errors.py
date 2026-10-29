"""Check scene time during timeline capture and the placement of offset calls."""

import pytest

from manim import Manager, Scene, tempconfig


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
