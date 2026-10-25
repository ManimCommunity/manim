"""Save timelines with the CLI and read them with the example reader."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = """from manim import *
class Sample(Scene):
    def construct(self):
        self.next_section("start")
        for _ in range(2):
            self.wait(.3, frozen_frame=False)
"""


def run(command, cwd, timeout=60):
    return subprocess.run(
        command,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=timeout,
    )


def run_cli(tmp_path, source, output, *extra):
    return run(
        [
            sys.executable,
            "-m",
            "manim",
            "--fps",
            "4",
            "-r",
            "32,16",
            "--media_dir",
            str(tmp_path / "media"),
            "--timeline-output",
            str(output),
            str(source),
            *extra,
        ],
        tmp_path,
    )


def test_dry_run_saves_an_evaluation_timeline(tmp_path):
    source = tmp_path / "scene.py"
    source.write_text(SOURCE)
    output = tmp_path / "report.json"
    result = run_cli(tmp_path, source, output, "--dry_run")
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(output.read_text())
    assert data["mode"] == "evaluate"
    assert data["duration"] == 1
    assert data["source"]["path"] == "scene.py"
    assert not list((tmp_path / "media").rglob("*.mp4"))


def test_render_saves_a_video_and_its_timeline(tmp_path):
    source = tmp_path / "scene.py"
    source.write_text(SOURCE)
    output = tmp_path / "report.json"
    result = run_cli(tmp_path, source, output)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(output.read_text())
    assert data["mode"] == "render"
    assert [entry["kind"] for entry in data["entries"]] == ["section", "wait", "wait"]
    assert list((tmp_path / "media").rglob("Sample.mp4"))


def test_render_with_a_timeline_runs_the_scene_s_own_render(tmp_path):
    source = tmp_path / "scene.py"
    source.write_text(
        SOURCE
        + """
    def render(self, *args, **kwargs):
        rerun = super().render(*args, **kwargs)
        Path("override-ran").write_text("yes")
        return rerun
"""
    )
    source.write_text("from pathlib import Path\n" + source.read_text())
    output = tmp_path / "report.json"
    result = run_cli(tmp_path, source, output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(output.read_text())["mode"] == "render"
    assert (tmp_path / "override-ran").read_text() == "yes"


def test_reader_prints_entries_and_detects_source_changes(tmp_path):
    source = tmp_path / "scene.py"
    source.write_text(SOURCE)
    output = tmp_path / "report.json"
    assert run_cli(tmp_path, source, output, "--dry_run").returncode == 0
    html = tmp_path / "view.html"
    command = [
        sys.executable,
        "-S",
        str(ROOT / "examples/timeline_reader.py"),
        str(output),
        "--source-root",
        str(tmp_path),
        "--html",
        str(html),
    ]
    reader = run(command, tmp_path)
    assert reader.returncode == 0, reader.stderr
    assert "source unchanged" in reader.stdout
    assert "wait 0.000–0.500s · scene.py:6" in reader.stdout
    assert "section &#x27;start&#x27;" in html.read_text()
    source.write_text(SOURCE + "\n# edited\n")
    assert "source changed" in run(command, tmp_path).stdout


def test_failure_keeps_the_previous_report_and_batches_are_rejected(tmp_path):
    source = tmp_path / "scene.py"
    output = tmp_path / "report.json"
    source.write_text(SOURCE)
    assert run_cli(tmp_path, source, output, "--dry_run").returncode == 0
    previous = output.read_bytes()
    source.write_text(SOURCE + '        raise RuntimeError("broken")\n')
    failed = run_cli(tmp_path, source, output, "--dry_run")
    assert failed.returncode != 0
    assert "broken" in failed.stdout + failed.stderr
    assert output.read_bytes() == previous
    source.write_text(SOURCE + "\nclass Another(Scene):\n    pass\n")
    result = run_cli(tmp_path, source, output, "--dry_run", "Sample", "Another")
    assert result.returncode != 0
    assert "exactly one" in result.stdout + result.stderr
    assert output.read_bytes() == previous
