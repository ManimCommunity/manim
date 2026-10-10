"""Validate captured clocks, event completion, and declaration placement offsets."""

import pytest

from manim import Manager, Scene, tempconfig
from tests.helpers.audio import write_wav


@pytest.fixture
def tone(tmp_path):
    return str(write_wav(tmp_path / "tone.wav", seconds=0.1))


def test_backwards_observed_time_cannot_publish():
    class Rewinds(Scene):
        def construct(self):
            self.wait(1, frozen_frame=False)
            self.renderer.time = 0.5
            self.wait(1, frozen_frame=False)

    with tempconfig({"format": "none", "frame_rate": 4, "progress_bar": "none"}):
        manager = Manager(Rewinds())
        with pytest.raises(ValueError, match="backwards"):
            manager.evaluate(capture_timeline=True)
        with pytest.raises(RuntimeError, match="No completed"):
            manager.timeline


def test_declaration_placement_offsets_are_not_clock_rewinds(tone):
    class Offsets(Scene):
        def construct(self):
            self.wait(1, frozen_frame=False)
            self.add_sound(tone, time_offset=-0.5)
            self.add_sound(tone, time_offset=2)
            self.add_subcaption("past", offset=-1)
            self.wait(1, frozen_frame=False)

    with tempconfig({"format": "none", "frame_rate": 4, "progress_bar": "none"}):
        manager = Manager(Offsets())
        manager.evaluate(capture_timeline=True)
    data = manager.timeline.to_dict()
    assert data["end"] == 2
    assert [item["start"] for item in data["declarations"]] == [0.5, 3, 0]
