"""Draw requested frames during evaluation."""

import numpy as np
import pytest

from manim import (
    RIGHT,
    UP,
    Circle,
    Dot,
    Manager,
    Scene,
    Square,
    tempconfig,
)
from manim.scene.scene_file_writer import SceneFileWriter

FRAME_RATE = 4


class Motion(Scene):
    def construct(self):
        self.add(Circle())
        square = Square()
        dot = Dot().add_updater(lambda dot: dot.next_to(square, UP))
        self.add(square, dot)
        self.play(square.animate.shift(RIGHT), run_time=0.6)
        self.wait(0.5, frozen_frame=True)
        dot.clear_updaters()
        dot.add_updater(lambda dot, dt: dot.shift(dt * RIGHT))
        self.wait(0.5, frozen_frame=False)
        start = self.time
        self.wait(2, stop_condition=lambda: self.time >= start + 0.5)


def settings(backend, tmp_path, output="none"):
    return {
        "renderer": backend,
        "format": output,
        "frame_rate": FRAME_RATE,
        "pixel_width": 64,
        "pixel_height": 36,
        "live_preview": False,
        "disable_caching": True,
        "progress_bar": "none",
        "media_dir": str(tmp_path / "media"),
    }


@pytest.mark.parametrize("backend", ["cairo", "opengl"])
def test_frames_match_the_rendered_frames(backend, tmp_path, monkeypatch):
    rendered = []
    original = SceneFileWriter.write_frame

    def write_frame(writer, pixels, *, repeat=1):
        rendered.extend([pixels.copy()] * repeat)
        original(writer, pixels, repeat=repeat)

    monkeypatch.setattr(SceneFileWriter, "write_frame", write_frame)
    with tempconfig(settings(backend, tmp_path, output="mp4")):
        Manager(Motion()).render()
    assert len(rendered) == 3 + 2 + 2 + 2

    frames = []
    with tempconfig(settings(backend, tmp_path)):
        manager = Manager(Motion())
        times = [(index + 0.5) / FRAME_RATE for index in range(len(rendered))]
        manager.evaluate(frame_times=times, on_frame=frames.append)
    assert [frame.frame_index for frame in frames] == list(range(len(rendered)))
    for frame, expected in zip(frames, rendered, strict=True):
        np.testing.assert_array_equal(np.asarray(frame.image), expected)


def test_frames_are_delivered_in_time_order(tmp_path):
    class Short(Scene):
        def construct(self):
            self.play(Square().animate.shift(RIGHT), run_time=0.3)
            self.wait(1, frozen_frame=True)

    frames = []
    with tempconfig(settings("cairo", tmp_path)):
        manager = Manager(Short())
        manager.evaluate(
            frame_times=[1.0, 0.3, 0.25, 0.3, 1.5, 99],
            on_frame=frames.append,
        )
    assert [
        (frame.requested_time, frame.frame_index, frame.time) for frame in frames
    ] == [(0.25, 1, 0.25), (0.3, 1, 0.25), (0.3, 1, 0.25), (1.0, 4, 1.0)]
    # Frames are drawn with a temporary renderer; evaluation opens no resources.
    assert manager._file_writer is None
    assert manager.renderer._target is None


def test_invalid_requests_are_rejected(tmp_path):
    with tempconfig(settings("cairo", tmp_path)):
        for times in ([-1], [float("nan")]):
            with pytest.raises(ValueError, match="finite"):
                Manager(Motion()).evaluate(frame_times=times, on_frame=print)
        with pytest.raises(ValueError, match="on_frame"):
            Manager(Motion()).evaluate(frame_times=[0])


def test_an_error_in_on_frame_ends_the_evaluation(tmp_path):
    def fail(frame):
        raise KeyError("stop")

    with tempconfig(settings("cairo", tmp_path)):
        manager = Manager(Motion())
        with pytest.raises(KeyError, match="stop"):
            manager.evaluate(frame_times=[0.5], on_frame=fail)
        assert manager.scene.time == 0.5
