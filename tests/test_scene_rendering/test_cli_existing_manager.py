"""Scenes rendered by one CLI run are independent and clean up after failures."""

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
        # Bound hangs while allowing cold imports and driver startup on shared CI.
        timeout=60,
    )


@pytest.mark.parametrize("backend", ["cairo", "opengl"])
def test_scenes_rendered_in_one_run_do_not_share_state(tmp_path, backend):
    result = render(
        tmp_path,
        backend,
        """from manim import RIGHT, RendererType, Scene, Square, config
class First(Scene):
    def construct(self):
        assert self.time == 0
        frame = self.camera.frame if config.renderer == RendererType.CAIRO else self.camera
        assert frame.get_center()[0] == 0
        frame.shift(RIGHT)
        self.add(Square())
        self.wait(0.25)
class Second(First):
    pass
""",
        "First",
        "Second",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list((tmp_path / "media").rglob("*.png"))) == 2


def test_failed_constructor_play_does_not_strand_encoder(tmp_path):
    result = render(
        tmp_path,
        "cairo",
        """from manim import Animation, Scene, Square
class BrokenAnimation(Animation):
    def interpolate_mobject(self, alpha):
        raise RuntimeError("constructor play failed")
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
    assert not list((tmp_path / "media").rglob("*.mp4"))
