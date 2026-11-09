"""Record timelines during evaluation and rendering, and save them as JSON."""

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from manim import Manager, Scene, Timeline, tempconfig
from manim.utils.module_ops import get_module

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
        options = {"levels": [1]}
        self.add_sound("missing.wav", time_offset=.25, gain=-3, settings=options)
        options["levels"].append(2)
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
    with tempconfig(
        {**SETTINGS, "format": "none", "media_dir": str(tmp_path / "media")}
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
    assert data["duration"] == scene.time == 2.5
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
    assert (sound["file"], sound["start"], sound["gain"]) == ("missing.wav", 2.75, -3)
    assert sound["duration"] is None
    # Later changes to the arguments are not recorded.
    assert sound["options"] == {"settings": {"levels": [1]}}

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


def test_reference_scene_matches_the_stored_format():
    path = ROOT / "tests" / "control_data" / "timeline_scene.py"
    with tempconfig({**SETTINGS, "format": "none"}):
        manager = Manager(get_module(path).TimelineFormat())
        manager.evaluate(capture_timeline=True)
    data = manager.timeline.to_dict()
    expected = json.loads(
        (ROOT / "tests" / "control_data" / "timeline-v1.json").read_text()
    )
    # The hash changes with every edit of the scene file.
    data["source"]["sha256"] = expected["source"]["sha256"]
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


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_a_written_timeline_has_the_permissions_of_a_new_file(scene_class, tmp_path):
    manager = Manager(scene_class())
    manager.evaluate(capture_timeline=True)
    previous = os.umask(0o022)
    try:
        manager.timeline.write(tmp_path / "report.json")
    finally:
        os.umask(previous)
    assert stat.S_IMODE((tmp_path / "report.json").stat().st_mode) == 0o644


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


def test_source_is_hashed_when_the_run_starts(scene_class, tmp_path):
    path = tmp_path / "scene.py"
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    scene = scene_class()
    scene.add_updater(lambda dt: path.write_text("# edited during the run\n"))
    manager = Manager(scene)
    manager.evaluate(capture_timeline=True)
    assert path.read_text() == "# edited during the run\n"
    assert manager.timeline.to_dict()["source"]["sha256"] == expected


def _guide_blocks(language):
    """Return the code blocks of the timeline guide in the given language."""
    guide = ROOT / "docs" / "source" / "guides" / "timeline.rst"
    text = guide.read_text(encoding="utf-8")
    blocks = []
    for part in text.split(f".. code-block:: {language}\n\n")[1:]:
        lines = []
        for line in part.splitlines():
            if line and not line.startswith("    "):
                break
            lines.append(line[4:])
        blocks.append("\n".join(lines).strip() + "\n")
    return blocks


def test_guide_example_prints_and_saves_what_the_guide_shows(tmp_path, monkeypatch):
    example, reader = _guide_blocks("python")
    (tmp_path / "example.py").write_text(example, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    with tempconfig({**SETTINGS, "format": "none"}):
        result = subprocess.run(
            [sys.executable, "example.py"],
            capture_output=True,
            encoding="utf-8",
            timeout=120,
        )
    assert result.returncode == 0, result.stderr
    printed = [line for line in result.stdout.splitlines() if line.strip()]
    assert printed[-4:] == _guide_blocks("text")[0].splitlines()

    saved = json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    shown = json.loads(_guide_blocks("json")[0].replace('"…"', '"hash"'))
    saved["source"]["sha256"] = "hash"
    assert saved == shown

    reader_output = subprocess.run(
        [sys.executable, "-c", reader],
        capture_output=True,
        encoding="utf-8",
        timeout=60,
    )
    assert reader_output.returncode == 0, reader_output.stderr
    assert reader_output.stdout.split() == [
        "section",
        "6",
        "play",
        "7",
        "caption",
        "7",
        "wait",
        "8",
    ]


def test_a_handled_play_failure_leaves_no_entry():
    from manim import Animation, Square

    class FailsHalfway(Animation):
        def interpolate_mobject(self, alpha):
            if alpha > 0.5:
                raise RuntimeError("failed halfway")

    class Handled(Scene):
        def construct(self):
            self.wait(0.25)
            with pytest.raises(RuntimeError, match="failed halfway"):
                self.play(FailsHalfway(Square()), run_time=0.5)
            self.wait(0.25)

    def entries(run):
        with tempconfig({**SETTINGS, "format": "none"}):
            manager = Manager(Handled())
            run(manager)
        return [
            (entry["kind"], entry["start"], entry["end"])
            for entry in manager.timeline.to_dict()["entries"]
        ]

    rendered = entries(lambda manager: manager.render(capture_timeline=True))
    assert rendered == [("wait", 0, 0.25), ("wait", 0.25, 0.5)]
    evaluated = entries(lambda manager: manager.evaluate(capture_timeline=True))
    assert [entry[0] for entry in evaluated] == ["wait", "wait"]


def test_a_render_captures_a_timeline_when_the_manager_is_told_to(tmp_path):
    # For renders started by Scene.render(), which a scene may override without
    # passing capture_timeline on, as the command line's --timeline-output does.
    class Overridden(Scene):
        def construct(self):
            self.wait(0.25)

        def render(self, preview=False):
            return super().render(preview)

    with tempconfig({"format": "none", "media_dir": str(tmp_path), "frame_rate": 4}):
        scene = Overridden()
        manager = scene._get_manager()
        manager.capture_timeline = True
        scene.render()
    assert [entry["kind"] for entry in manager.timeline.to_dict()["entries"]] == [
        "wait"
    ]
