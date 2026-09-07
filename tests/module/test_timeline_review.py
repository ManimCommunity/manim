"""Adversarial regressions for completed timeline observations."""

import contextlib

import pytest

from manim import Manager, Scene, Wait, tempconfig


@pytest.mark.parametrize("location", ["between-events", "after-events", "during-event"])
def test_backwards_observed_time_cannot_publish(location):
    class Rewinds(Scene):
        def construct(self):
            self.wait(1, frozen_frame=False)
            if location == "during-event":
                self.add_updater(lambda dt: self.add_subcaption("sample"))
                self.wait(1, frozen_frame=False)
            else:
                self.renderer.time = 0.5
                if location == "between-events":
                    self.wait(1, frozen_frame=False)

        def update_to_time(self, t):
            super().update_to_time(t)
            if location == "during-event" and self.time >= 1:
                self.renderer.time = 0.5
                self.add_subcaption("rewound")

    with tempconfig({"format": "none", "frame_rate": 4, "progress_bar": "none"}):
        manager = Manager(Rewinds())
        with pytest.raises(ValueError, match="backwards"):
            manager.evaluate(capture_timeline=True)
        with pytest.raises(RuntimeError, match="No completed"):
            manager.timeline


def test_private_sample_loop_cannot_publish_unrecorded_time():
    class BypassesEntry(Scene):
        def construct(self):
            self.compile_animation_data(Wait(1, frozen_frame=False))
            self.begin_animations()
            self.manager._play_internal()

    with tempconfig({"format": "none", "frame_rate": 4, "progress_bar": "none"}):
        manager = Manager(BypassesEntry())
        with pytest.raises(RuntimeError, match="outside a timed event"):
            manager.evaluate(capture_timeline=True)


def test_declaration_placement_offsets_are_not_clock_rewinds():
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
    assert data["end"] == 2
    assert [item["start"] for item in data["declarations"]] == [0.5, 3, 0]


def test_caught_unserializable_declaration_does_not_publish_partial_capture():
    class Unsupported(Scene):
        def construct(self):
            with contextlib.suppress(TypeError):
                self.add_sound("missing.wav", custom=object())
            self.wait(1, frozen_frame=False)

    with tempconfig({"format": "none", "frame_rate": 4, "progress_bar": "none"}):
        manager = Manager(Unsupported())
        with pytest.raises(RuntimeError, match="incomplete"):
            manager.evaluate(capture_timeline=True)
        with pytest.raises(RuntimeError, match="No completed"):
            manager.timeline
