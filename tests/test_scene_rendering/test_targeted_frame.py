"""Check frame selection against actual writer delivery on Cairo and OpenGL."""

import math
from unittest.mock import Mock

import numpy as np
import pytest

from manim import RIGHT, Manager, Scene, Square, linear, tempconfig


@pytest.fixture(params=["cairo", "opengl"])
def settings(request, tmp_path):
    with tempconfig(
        {
            "renderer": request.param,
            "format": "mp4",
            "frame_rate": 4,
            "pixel_width": 96,
            "pixel_height": 64,
            "live_preview": False,
            "disable_caching": True,
            "progress_bar": "none",
            "media_dir": str(tmp_path),
        }
    ):
        yield


class MovingScene(Scene):
    def setup(self):
        self.trace = []
        self.square = Square(fill_opacity=1)
        self.add(self.square)

    def construct(self):
        self.play(self.square.animate.shift(RIGHT), run_time=0.3, rate_func=linear)
        self.trace.append("move")
        self.wait(0.75, frozen_frame=True)
        self.trace.append("hold")
        start = self.time
        self.add_updater(lambda dt: self.square.shift(dt * RIGHT))
        self.wait(1, stop_condition=lambda: self.time >= start + 0.5)
        self.trace.append("end")

    def tear_down(self):
        self.trace.append("tear_down")
        self.clear()


@pytest.mark.parametrize(
    ("timestamp", "index", "completed"),
    [(0.25, 1, []), (0.8, 3, ["move"]), (1.25, 5, ["move", "hold"])],
)
def test_capture_matches_writer_before_finish_and_teardown(
    settings, monkeypatch, timestamp, index, completed
):
    reference = MovingScene()
    writer = Manager(reference).file_writer
    write = writer.write_frame
    frames = []

    def record(frame, *, repeat=1):
        frames.extend(
            (frame.copy(), reference.square.get_center().copy()) for _ in range(repeat)
        )
        write(frame, repeat=repeat)

    monkeypatch.setattr(writer, "write_frame", record)
    reference.render()
    assert len(frames) == 7
    scene = MovingScene()
    manager = Manager(scene)
    monkeypatch.setattr(
        scene.renderer, "_file_writer_class", Mock(side_effect=AssertionError)
    )
    with tempconfig({"disable_caching": False, "from_animation_number": 10}):
        result = manager.capture_frame_at(timestamp)
    assert (result.requested_time, result.time, result.frame_index) == (
        timestamp,
        index / 4,
        index,
    )
    np.testing.assert_array_equal(result.image, frames[index][0])
    np.testing.assert_array_equal(scene.square.get_center(), frames[index][1])
    assert scene.time == (index + 1) / 4
    assert scene.trace == [*completed, "tear_down"]
    assert not scene.mobjects
    assert manager._file_writer is None
    assert manager._closed
    assert scene.renderer._closed


def test_final_endpoint_is_a_miss(settings):
    manager = Manager(MovingScene())
    assert manager.capture_frame_at(1.75) is None
    assert manager.scene.trace == ["move", "hold", "end", "tear_down"]
    assert manager._closed
    assert manager._file_writer is None


@pytest.mark.parametrize("settings", ["cairo"], indirect=True)
@pytest.mark.parametrize(
    ("timestamp", "index"), [(math.nextafter(15 / 22, 0), 14), (15 / 22, 15)]
)
def test_frozen_hold_selects_exact_float_boundary(settings, timestamp, index):
    class Hold(Scene):
        def construct(self):
            self.wait(1, frozen_frame=True)

    # 15 / 22 * 22 rounds below 15; compare interval boundaries directly.
    with tempconfig({"frame_rate": 22}):
        result = Manager(Hold()).capture_frame_at(timestamp)
    assert result.frame_index == index
    assert result.time == index / 22


@pytest.mark.parametrize("settings", ["cairo"], indirect=True)
@pytest.mark.parametrize("timestamp", [-1, math.nan, math.inf])
def test_invalid_timestamp_leaves_scene_unused(settings, timestamp):
    with Manager(Scene()) as manager:
        with pytest.raises(ValueError, match="finite and nonnegative"):
            manager.capture_frame_at(timestamp)
        assert manager._frame_request is None
        assert manager._file_writer is None


def test_teardown_failure_closes_capture_resources(settings, monkeypatch):
    manager = Manager(MovingScene())
    error = KeyboardInterrupt("teardown failed")
    monkeypatch.setattr(manager.scene, "tear_down", Mock(side_effect=error))
    with pytest.raises(KeyboardInterrupt) as caught:
        manager.capture_frame_at(0.25)
    assert caught.value is error
    assert manager._closed
    assert manager.renderer._closed
    assert manager._file_writer is None
