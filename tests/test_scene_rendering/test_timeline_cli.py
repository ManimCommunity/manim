"""No-media JSON production consumed by an independent stdlib-only reader."""

import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = """from manim import *
class Sample(Scene):
    def construct(self):
        self.next_section("start")
        for _ in range(2):
            self.wait(.3, frozen_frame=False)
        self.add_sound("missing.wav")
"""


def run_cli(tmp_path, source, output, *extra):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "manim",
            "--fps",
            "4",
            "--media_dir",
            str(tmp_path / "media"),
            "--timeline-output",
            str(output),
            *extra,
            str(source),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("backend", ["cairo", "opengl"])
def test_cli_and_independent_reader(tmp_path, backend):
    source = tmp_path / "scene.py"
    source.write_text(SOURCE)
    output = tmp_path / "report.json"
    result = run_cli(tmp_path, source, output, "--renderer", backend)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(output.read_text())
    assert data["end"] == 1
    assert data["source"]["path"] == "scene.py"
    assert "before-loading" in data["source"]["provenance"]
    assert not (tmp_path / "media").exists()
    assert "output" not in data

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
    reader = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert reader.returncode == 0, reader.stderr
    assert "event-000000 wait" in reader.stdout
    assert "source matches captured bytes" in reader.stdout
    assert "event-000001" in html.read_text()
    source.write_text(SOURCE + "\n# edited\n")
    stale = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert "source STALE" in stale.stdout


def test_source_change_during_retained_scope_cleanup_prevents_publication(tmp_path):
    source = tmp_path / "scene.py"
    source.write_text("""from pathlib import Path
from manim import Manager, Scene
class ClosingManager(Manager):
    def close(self):
        super().close()
        path = Path(__file__)
        path.write_text(path.read_text() + "\\n# changed\\n")
class Sample(Scene):
    def __init__(self):
        super().__init__()
        ClosingManager(self)
    def construct(self):
        self.wait(1, frozen_frame=False)
""")
    output = tmp_path / "report.json"
    output.write_text("previous")
    result = run_cli(tmp_path, source, output)
    assert result.returncode != 0
    assert "source changed" in result.stdout + result.stderr
    assert output.read_text() == "previous"


def test_cli_executes_captured_primary_bytes_not_stale_bytecode(tmp_path):
    source = tmp_path / "scene.py"
    old = "from manim import *\nclass Sample(Scene):\n    def construct(self):\n        self.wait(1, frozen_frame=False)\n"
    source.write_text(old)
    stamp = source.stat()
    py_compile.compile(str(source), doraise=True)
    source.write_text(old.replace("wait(1", "wait(2"))
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    output = tmp_path / "report.json"
    result = run_cli(tmp_path, source, output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(output.read_text())["end"] == 2


def test_relative_output_is_anchored_before_user_code_changes_directory(tmp_path):
    source = tmp_path / "scene.py"
    destination = tmp_path / "nested"
    destination.mkdir()
    source.write_text(
        "import os\n"
        + SOURCE.replace(
            'self.next_section("start")', f"os.chdir({str(destination)!r})"
        )
    )
    result = run_cli(tmp_path, source, Path("report.json"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "report.json").is_file()
    assert not (destination / "report.json").exists()


def test_cli_failure_preserves_previous_report_and_rejects_batches(tmp_path):
    source = tmp_path / "scene.py"
    output = tmp_path / "report.json"
    source.write_text(SOURCE)
    assert run_cli(tmp_path, source, output).returncode == 0
    previous = output.read_bytes()
    source.write_text(
        SOURCE.replace('self.add_sound("missing.wav")', 'raise RuntimeError("broken")')
    )
    assert run_cli(tmp_path, source, output).returncode != 0
    assert output.read_bytes() == previous
    source.write_text(SOURCE + "\nclass Another(Scene):\n    pass\n")
    result = run_cli(tmp_path, source, output)
    assert result.returncode != 0
    assert "exactly one" in result.stdout + result.stderr
    assert output.read_bytes() == previous
