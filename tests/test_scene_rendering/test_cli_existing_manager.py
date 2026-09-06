"""The CLI must reuse Managers attached by explicit constructor-time requests."""

import sys

import pytest

from manim import capture


@pytest.mark.parametrize("renderer", ["cairo", "opengl"])
@pytest.mark.parametrize("override", [False, True])
def test_cli_retires_batch_resources_including_render_overrides(
    tmp_path, renderer, override
):
    marker = tmp_path / "retired"
    script = tmp_path / "batch.py"
    script.write_text(
        "import atexit\n"
        "from pathlib import Path\n"
        "from manim import Scene, Square\n"
        "instances = []\n"
        "def check():\n"
        "    assert len(instances) == 2\n"
        "    assert all(s.manager._closed for s in instances)\n"
        "    assert all(s.renderer._closed for s in instances)\n"
        f"    Path({str(marker)!r}).write_text('retired')\n"
        "atexit.register(check)\n"
        "class First(Scene):\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        super().__init__(*args, **kwargs)\n"
        "        instances.append(self)\n"
        "    def construct(self):\n"
        "        self.add(Square())\n"
        + (
            "    def render(self, preview=False):\n"
            "        self.construct()\n"
            "        self.renderer.update_frame(self)\n"
            "        return False\n"
            if override
            else ""
        )
        + "class Second(First):\n    pass\n",
        encoding="utf-8",
    )
    out, err, code = capture(
        [
            sys.executable,
            "-m",
            "manim",
            "--renderer",
            renderer,
            "--format",
            "png",
            "-r",
            "64,32",
            "--media_dir",
            str(tmp_path / "media"),
            "--write_all",
            str(script),
        ]
    )
    assert code == 0, out + err
    assert marker.exists(), out + err


@pytest.mark.parametrize("renderer", ["cairo", "opengl"])
@pytest.mark.parametrize("operation", ["self.renderer.file_writer", "self.get_image()"])
def test_cli_reuses_manager_after_constructor_request(tmp_path, renderer, operation):
    script = tmp_path / "inspect_constructor.py"
    script.write_text(
        "from manim import Scene, Square\n"
        "class InspectConstructor(Scene):\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        super().__init__(*args, **kwargs)\n"
        f"        {operation}\n"
        "    def construct(self):\n"
        "        self.add(Square())\n",
        encoding="utf-8",
    )
    media = tmp_path / "media"
    out, err, code = capture(
        [
            sys.executable,
            "-m",
            "manim",
            "--renderer",
            renderer,
            "--format",
            "png",
            "-r",
            "64,32",
            "--media_dir",
            str(media),
            str(script),
            "InspectConstructor",
        ]
    )
    assert code == 0, out + err
    assert len(list(media.rglob("*.png"))) == 1
