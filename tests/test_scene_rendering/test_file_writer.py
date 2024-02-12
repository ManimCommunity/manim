import sys

import pytest

from manim import Scene, Create, Star, tempconfig

from manim.utils.commands import capture
from tests.utils.video_tester import video_comparison


class StarScene(Scene):
    def construct(self):
        star = Star()
        self.play(Create(star))

@video_comparison("H264Codec.json", "videos/480p15/H264Codec.mp4")
@pytest.mark.slow
def test_h264_codec(tmp_path):
    with tempconfig({
        "media_dir": tmp_path,
        "quality": "low_quality",
        "output_file": "H264Codec",
    }):
        StarScene().render()

@video_comparison("qtrleCodec.json", "videos/480p15/qtrleCodec.mov")
@pytest.mark.slow
def test_qtrle_codec(tmp_path):
    with tempconfig({
        "media_dir": tmp_path,
        "quality": "low_quality",
        "transparent": True,
        "output_file": "qtrleCodec",
    }):
        StarScene().render()


@video_comparison("vp9Codec.json", "videos/480p15/vp9Codec.webm")
@pytest.mark.slow
def test_vp9_codec(tmp_path):
    with tempconfig({
        "media_dir": tmp_path,
        "quality": "low_quality",
        "format": "webm",
        "transparent": True,
        "output_file": "vp9Codec",
    }):
        StarScene().render()


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
