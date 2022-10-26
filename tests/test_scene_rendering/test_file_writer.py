import sys

import pytest

from manim.utils.commands import capture


@pytest.mark.slow
def test_unicode_partial_movie(tmpdir, simple_scenes_path):
    # Characters that failed for a user on Windows
    # due to its weird default encoding.
    unicode_str = "三角函数"

    scene_name = "SquareToCircle"
    command = [
        sys.executable,
        "-m",
        "manim",
        "--media_dir",
        str(tmpdir / unicode_str),
        str(simple_scenes_path),
        scene_name,
    ]

    _, err, exit_code = capture(command)
    assert exit_code == 0, err
