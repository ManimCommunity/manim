import sys

import av
import numpy as np
import pytest

from manim import Create, Scene, Star, tempconfig
from manim.utils.commands import capture, get_video_metadata
from tests.utils.video_tester import video_comparison


class StarScene(Scene):
    def construct(self):
        star = Star()
        self.play(Create(star))


@pytest.mark.slow
@pytest.mark.parametrize(
    "format, transparent, codec, pixel_format",
    [
        ("mp4", False, "h264", "yuv420p"),
        ("mov", False, "h264", "yuv420p"),
        ("mov", True, "qtrle", "argb"),
        ("webm", False, "vp9", "yuv420p"),
        ("webm", True, "vp9", "yuv420p"),
    ],
)
def test_codecs(tmp_path, format, transparent, codec, pixel_format):
    output_filename = f"codec_{format}_{'transparent' if transparent else 'opaque'}"
    with tempconfig(
        {
            "media_dir": tmp_path,
            "quality": "low_quality",
            "format": format,
            "transparent": transparent,
            "output_file": output_filename,
        }
    ):
        StarScene().render()

    video_path = tmp_path / "videos" / "480p15" / f"{output_filename}.{format}"
    assert video_path.exists()

    metadata = get_video_metadata(video_path)
    assert metadata == {
        "width": 854,
        "height": 480,
        "nb_frames": "15",
        "duration": "1.000000",
        "avg_frame_rate": "15/1",
        "codec_name": codec,
        "pix_fmt": pixel_format,
    }

    with av.open(video_path) as container:
        first_frame = next(container.decode(video=0)).to_ndarray(format="rgba")
        target_rgba = (
            np.array([0, 0, 0, 255]) if not transparent else np.array([0, 0, 0, 0])
        )
        np.testing.assert_array_equal(first_frame[0, 0], target_rgba)


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
