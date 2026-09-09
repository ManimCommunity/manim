"""Each CLI run has a fresh Scene and reuses that scene's manager."""

import os
import subprocess
import sys

import pytest


def render(tmp_path, backend, source, *selection, output="png"):
    script = tmp_path / "scene.py"
    script.write_text(source, encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "manim",
            "--renderer",
            backend,
            "--format",
            output,
            "-r",
            "64,32",
            "--fps",
            "4",
            "--disable_caching",
            "--progress_bar=none",
            "--media_dir",
            str(tmp_path / "media"),
            str(script),
            *selection,
        ],
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=20,
    )


@pytest.mark.parametrize("backend", ["cairo", "opengl"])
@pytest.mark.parametrize("selection", [("First", "Second"), ("-a",)])
def test_batch_and_rerun_start_fresh_and_retire_each_owner(
    tmp_path, backend, selection
):
    result = render(
        tmp_path,
        backend,
        """import atexit
from pathlib import Path
from manim import Scene, Square, RIGHT, config, RendererType
instances = []
def check():
    assert [type(s).__name__ for s in instances] == ["First", "Second", "Second"]
    assert all(s.manager._closed and s.renderer._closed for s in instances)
    for owners in ([s.renderer for s in instances], [s.camera for s in instances],
                   [s.manager.file_writer for s in instances]):
        assert len({id(owner) for owner in owners}) == 3
    Path(__file__).with_suffix(".retired").touch()
atexit.register(check)
class First(Scene):
    def __init__(self, *args, **kwargs):
        if instances:
            assert instances[-1].manager._closed
            assert instances[-1].renderer._closed
        super().__init__(*args, **kwargs)
        self.renderer.file_writer  # Attaches a Manager before the CLI gets the Scene.
        instances.append(self)
    def construct(self):
        assert self.time == self.renderer.num_plays == 0
        frame = self.camera.frame if config.renderer == RendererType.CAIRO else self.camera
        assert frame.get_center()[0] == 0
        frame.shift(RIGHT)
        self.add(Square())
class Second(First):
    def render(self, preview=False):
        # Deliberately bypass Manager.render; the CLI must still retire the scope.
        self.construct()
        self.renderer.update_frame(self)
        return sum(type(s) is Second for s in instances) == 1
""",
        *selection,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "scene.retired").is_file(), result.stdout + result.stderr
    assert len(list((tmp_path / "media").rglob("*.png"))) == 1


@pytest.mark.parametrize("backend", ["cairo", "opengl"])
def test_cli_accepts_constructor_snapshot(tmp_path, backend):
    result = render(
        tmp_path,
        backend,
        """from manim import Scene, Square
class InspectConstructor(Scene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_image()
    def construct(self):
        self.add(Square())
""",
        "InspectConstructor",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list((tmp_path / "media").rglob("*.png"))) == 1


@pytest.mark.parametrize("backend", ["cairo", "opengl"])
def test_failed_constructor_play_does_not_strand_encoder(
    tmp_path, backend, monkeypatch
):
    # The helper must use UTF-8 even when the parent's stream encoding differs.
    monkeypatch.setenv("PYTHONIOENCODING", "ascii")
    result = render(
        tmp_path,
        backend,
        """from manim import Animation, Scene, Square
class BrokenAnimation(Animation):
    def interpolate_mobject(self, alpha):
        raise RuntimeError("constructor play failed → cleanup")
class BrokenConstructor(Scene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.play(BrokenAnimation(Square()))
""",
        "BrokenConstructor",
        output="mp4",
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "constructor play failed" in result.stdout + result.stderr
    assert "→ cleanup" in result.stdout + result.stderr
    assert not list((tmp_path / "media").rglob("*.mp4"))
