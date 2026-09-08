"""Capture evaluation steps and atomically save immutable reports."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from manim import Manager, Scene, Timeline, tempconfig
from manim.utils.module_ops import get_module

SOURCE = """from manim import *
class TimelineFixture(Scene):
    def construct(self):
        self.observed = []
        self.add_updater(lambda dt: self.observed.append((self.time, dt)))
        self.next_section("opening", skip_animations=True)
        self.play(Square().animate.shift(RIGHT), run_time=.3, subcaption="move")
        for _ in range(2):
            self.wait(.3, frozen_frame=False)
        self.wait(.3, frozen_frame=True)
        start = self.time
        self.wait(1, stop_condition=lambda: self.time >= start + .5)
        options = {"levels": [1]}
        self.add_sound("missing.wav", time_offset=.25, gain=-3, settings=options)
        options["levels"].append(2)
"""


@pytest.fixture(params=["cairo", "opengl"])
def fixture_scene(request, tmp_path):
    path = tmp_path / "scene.py"
    path.write_text(SOURCE)
    with tempconfig(
        {
            "renderer": request.param,
            "format": "none",
            "frame_rate": 4,
            "pixel_width": 32,
            "pixel_height": 16,
            "live_preview": False,
            "progress_bar": "none",
            "media_dir": str(tmp_path / "media"),
        }
    ):
        yield get_module(path).TimelineFixture, path


@pytest.fixture
def snapshot():
    path = Path(__file__).parents[1] / "control_data/timeline-v1.json"
    return Timeline.from_json(path.read_text())


def test_capture_observes_actual_schedule_without_changing_evaluation(
    fixture_scene, monkeypatch
):
    scene_class, _ = fixture_scene
    plain = scene_class()
    Manager(plain).evaluate()
    scene = scene_class()
    manager = Manager(scene)
    reject = Mock(side_effect=AssertionError("raster/media requested"))
    monkeypatch.setattr(scene.renderer, "_file_writer_class", reject)
    monkeypatch.setattr(manager, "_draw_animation_frame", reject)
    with pytest.raises(RuntimeError, match="No completed"):
        manager.timeline
    manager.evaluate(capture_timeline=True)
    assert scene.observed == plain.observed
    assert scene.time == plain.time == 2.25
    data = manager.timeline.to_dict()
    assert data["complete"]
    assert data["policy"] == "no-raster-full"
    assert "output" not in data
    events = data["events"]
    assert len(events) == 5
    assert [event["kind"] for event in events] == [
        "play",
        "wait",
        "wait",
        "wait",
        "wait",
    ]
    assert [event["samples"] for event in events] == [2, 2, 2, 0, 2]
    assert events[3]["hold_intervals"] == 1
    assert events[4]["nominal_duration"] == 1
    assert events[4]["end"] - events[4]["start"] == 0.5
    assert events[1]["source"]["site_id"] == events[2]["source"]["site_id"]
    assert [events[i]["source"]["occurrence"] for i in (1, 2)] == [0, 1]
    section, caption, sound = data["declarations"]
    assert section["kind"] == "section"
    assert section["name"] == "opening"
    assert section["type"] == "default.normal"
    assert section["skip_requested"]
    assert caption["kind"] == "caption"
    assert (caption["content"], caption["start"], caption["end"]) == ("move", 0, 0.5)
    assert sound["kind"] == "sound"
    assert sound["start"] == 2.5
    assert sound["gain"] == -3
    assert sound["duration"] is None
    assert sound["options"] == {"settings": {"levels": [1]}}
    assert sound["asset"] == {"request": "missing.wav", "resolution": "unresolved"}
    assert manager._file_writer is None
    reject.assert_not_called()
    other = Manager(scene_class())
    other.evaluate(capture_timeline=True)
    assert other.timeline.to_json() == manager.timeline.to_json()


def test_snapshot_is_immutable_canonical_and_revision_checked(snapshot):
    document = snapshot.to_json()
    mutable = snapshot.to_dict()
    mutable["events"].clear()
    assert snapshot.to_json() == document
    assert Timeline.from_json(document).to_json() == document
    changed = json.loads(document)
    changed["end"] = 100
    with pytest.raises(ValueError, match="revision"):
        Timeline.from_json(json.dumps(changed))


def test_atomic_replace_failure_preserves_prior_json(snapshot, tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    path.write_text("previous")
    monkeypatch.setattr(
        Path, "replace", Mock(side_effect=OSError("publication failed"))
    )
    with pytest.raises(OSError, match="publication failed"):
        snapshot.write(path)
    assert path.read_text() == "previous"
    assert not list(tmp_path.glob(".report.json.*"))


def test_failed_capture_has_no_successful_snapshot(tmp_path):
    failure = KeyboardInterrupt("stop")

    class Broken(Scene):
        def construct(self):
            self.wait(1, frozen_frame=False)
            raise failure

    with tempconfig(
        {
            "format": "none",
            "frame_rate": 4,
            "pixel_width": 32,
            "pixel_height": 16,
            "progress_bar": "none",
        }
    ):
        manager = Manager(Broken())
        with pytest.raises(KeyboardInterrupt) as error:
            manager.evaluate(capture_timeline=True)
        assert error.value is failure
        with pytest.raises(RuntimeError, match="No completed"):
            manager.timeline
        assert manager._timeline_recorder is None


def test_compilation_captions_keep_event_order_and_rounded_placement():
    class Compiling(Scene):
        def compile_animation_data(self, *args, **kwargs):
            self.add_subcaption("during compilation", duration=1 / 3, offset=1 / 7)
            return super().compile_animation_data(*args, **kwargs)

        def construct(self):
            self.wait(1, frozen_frame=False)

    with tempconfig({"format": "none", "frame_rate": 4, "progress_bar": "none"}):
        manager = Manager(Compiling())
        manager.evaluate(capture_timeline=True)
    data = manager.timeline.to_dict()
    caption = data["declarations"][0]
    assert data["events"][0]["order"] < caption["order"]
    assert caption["event_id"] == data["events"][0]["id"]
    assert caption["content"] == "during compilation"
    assert caption["start"] == 0.142857
    assert caption["end"] == 0.476190


def test_generated_source_is_reported_as_unavailable():
    namespace = {"__name__": "unregistered_generated_scene", "Scene": Scene}
    exec(
        "class Generated(Scene):\n    def construct(self):\n        self.wait(1, frozen_frame=False)\n",
        namespace,
    )
    with tempconfig({"format": "none", "frame_rate": 4, "progress_bar": "none"}):
        manager = Manager(namespace["Generated"]())
        manager.evaluate(capture_timeline=True)
    data = manager.timeline.to_dict()
    assert data["source"]["coverage"] == "unavailable"
    assert data["events"][0]["source"]["coverage"] == "generated"
    assert data["events"][0]["source"]["site_id"] is None


def test_version_one_fixture_is_readable(snapshot):
    assert snapshot.to_dict()["end"] == 2.25
    assert len(snapshot.to_dict()["events"]) == 5


def test_changed_source_prevents_successful_capture(fixture_scene):
    scene_class, source = fixture_scene
    manager = Manager(scene_class())
    original = manager.tear_down

    def changed():
        original()
        source.write_text(source.read_text() + "\n# changed\n")

    manager.tear_down = changed
    with pytest.raises(RuntimeError, match="source changed"):
        manager.evaluate(capture_timeline=True)
    with pytest.raises(RuntimeError, match="No completed"):
        manager.timeline
