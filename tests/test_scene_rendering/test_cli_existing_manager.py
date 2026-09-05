"""The CLI must reuse Managers attached by explicit constructor-time requests."""

import sys

import pytest

from manim import capture


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
