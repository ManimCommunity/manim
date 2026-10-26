"""Record timelines during evaluation and rendering, and save them as JSON."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from manim import Manager, Scene, Timeline, tempconfig
from manim.utils.module_ops import get_module
from tests.helpers.audio import write_wav

ROOT = Path(__file__).resolve().parents[2]

SOURCE = """from manim import *
class TimelineFixture(Scene):
    def construct(self):
        self.observed = []
        self.add_updater(lambda dt: self.observed.append((self.time, dt)))
        self.next_section("opening")
        self.play(Square().animate.shift(RIGHT), run_time=.3, subcaption="move")
        self.next_section("skipped", skip_animations=True)
        for _ in range(2):
            self.wait(.3, frozen_frame=False)
        self.wait(.3, frozen_frame=True)
        start = self.time
        self.wait(1, stop_condition=lambda: self.time >= start + .5)
        self.add_sound("beep.wav", time_offset=.25, gain=-3, gain_to_background=-6)
"""

SETTINGS = {
    "renderer": "cairo",
    "frame_rate": 4,
    "pixel_width": 32,
    "pixel_height": 16,
    "live_preview": False,
    "progress_bar": "none",
}


@pytest.fixture
def scene_class(tmp_path):
    path = tmp_path / "scene.py"
    path.write_text(SOURCE)
    write_wav(tmp_path / "beep.wav", seconds=0.2)
    with tempconfig(
        {
            **SETTINGS,
            "format": "none",
            "media_dir": str(tmp_path / "media"),
            "assets_dir": str(tmp_path),
        }
    ):
        yield get_module(path).TimelineFixture


def test_evaluation_records_calls_without_rendering(scene_class, monkeypatch):
    plain = scene_class()
    Manager(plain).evaluate()
    scene = scene_class()
    manager = Manager(scene)
    reject = Mock(side_effect=AssertionError("rendering requested"))
    monkeypatch.setattr(scene.renderer, "_file_writer_class", reject)
    monkeypatch.setattr(scene.renderer, "render", reject)
    with pytest.raises(RuntimeError, match="No completed"):
        manager.timeline
    manager.evaluate(capture_timeline=True)
    reject.assert_not_called()
    assert scene.observed == plain.observed

    data = manager.timeline.to_dict()
    assert data["mode"] == "evaluate"
    assert data["scene"] == "TimelineFixture"
    assert data["frame_rate"] == 4
    assert data["duration"] == scene.time == 2.25
    assert not data["ended_early"]
    assert data["source"]["path"] == "scene.py"
    entries = data["entries"]
    assert [entry["kind"] for entry in entries] == [
        "section",
        "play",
        "caption",
        "section",
        "wait",
        "wait",
        "wait",
        "wait",
        "sound",
    ]
    play = entries[1]
    assert (play["start"], play["end"], play["run_time"]) == (0, 0.5, 0.3)
    assert play["source"] == {"path": "scene.py", "line": 7}
    assert not any(entry.get("skipped") for entry in entries)
    assert [entry.get("frozen") for entry in entries[4:8]] == [
        False,
        False,
        True,
        False,
    ]
    stopped = entries[7]
    assert (stopped["run_time"], stopped["end"] - stopped["start"]) == (1, 0.5)
    caption = entries[2]
    assert (caption["content"], caption["start"], caption["end"]) == ("move", 0, 0.5)
    assert entries[3]["skip_animations"]
    sound = entries[8]
    assert (sound["file"], sound["start"], sound["gain"]) == ("beep.wav", 2.5, -3)
    assert sound["duration"] == pytest.approx(0.2)
    assert sound["gain_to_background"] == -6

    other = Manager(scene_class())
    other.evaluate(capture_timeline=True)
    assert other.timeline.to_json() == manager.timeline.to_json()


def test_render_records_the_same_calls_and_marks_skipped_plays(scene_class, tmp_path):
    evaluated = Manager(scene_class())
    evaluated.evaluate(capture_timeline=True)
    with tempconfig({"format": "mp4", "disable_caching": True}):
        manager = Manager(scene_class())
        manager.render(capture_timeline=True)
    rendered = manager.timeline.to_dict()
    assert rendered["mode"] == "render"
    timed = [entry for entry in rendered["entries"] if "skipped" in entry]
    assert [entry["skipped"] for entry in timed] == [False, True, True, True, True]
    expected = evaluated.timeline.to_dict()
    for entry in rendered["entries"]:
        entry.pop("skipped", None)
    for entry in expected["entries"]:
        entry.pop("skipped", None)
    assert rendered["entries"] == expected["entries"]
    assert rendered["duration"] == expected["duration"]


def test_example_scene_matches_the_documented_format():
    path = ROOT / "example_scenes" / "timeline_scene.py"
    with tempconfig({**SETTINGS, "format": "none"}):
        manager = Manager(get_module(path).TimelineExample())
        manager.evaluate(capture_timeline=True)
    data = manager.timeline.to_dict()
    expected = json.loads(
        (ROOT / "tests" / "control_data" / "timeline-v1.json").read_text()
    )
    # The hash changes with every edit of the example file, and the sound's path
    # depends on where the repository is.
    data["source"]["sha256"] = expected["source"]["sha256"]
    [sound] = [entry for entry in data["entries"] if entry["kind"] == "sound"]
    [expected_sound] = [e for e in expected["entries"] if e["kind"] == "sound"]
    sound["file"] = expected_sound["file"]
    assert data == expected


def test_json_round_trip_and_version_check(scene_class):
    manager = Manager(scene_class())
    manager.evaluate(capture_timeline=True)
    timeline = manager.timeline
    document = timeline.to_json()
    copy = timeline.to_dict()
    copy["entries"].clear()
    assert timeline.to_json() == document
    assert Timeline.from_json(document).to_json() == document
    other = json.loads(document)
    other["version"] = 2
    with pytest.raises(ValueError, match="version 1"):
        Timeline.from_json(json.dumps(other))


def test_failed_write_keeps_the_previous_file(scene_class, tmp_path, monkeypatch):
    manager = Manager(scene_class())
    manager.evaluate(capture_timeline=True)
    path = tmp_path / "report.json"
    path.write_text("previous")
    monkeypatch.setattr(Path, "replace", Mock(side_effect=OSError("failed")))
    with pytest.raises(OSError, match="failed"):
        manager.timeline.write(path)
    assert path.read_text() == "previous"
    assert not list(tmp_path.glob(".report.json.*"))


def test_failed_run_records_no_timeline():
    failure = KeyboardInterrupt("stop")

    class Broken(Scene):
        def construct(self):
            self.wait(1, frozen_frame=False)
            raise failure

    with tempconfig({**SETTINGS, "format": "none"}):
        manager = Manager(Broken())
        with pytest.raises(KeyboardInterrupt) as error:
            manager.evaluate(capture_timeline=True)
        assert error.value is failure
        with pytest.raises(RuntimeError, match="No completed"):
            manager.timeline


def test_generated_code_has_no_source_location():
    namespace = {"__name__": "generated_scene", "Scene": Scene}
    exec(
        "class Generated(Scene):\n"
        "    def construct(self):\n"
        "        self.wait(1, frozen_frame=False)\n",
        namespace,
    )
    with tempconfig({**SETTINGS, "format": "none"}):
        manager = Manager(namespace["Generated"]())
        manager.evaluate(capture_timeline=True)
    data = manager.timeline.to_dict()
    assert data["source"] is None
    assert data["entries"][0]["source"] is None
