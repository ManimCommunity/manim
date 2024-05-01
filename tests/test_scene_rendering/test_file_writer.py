import sys

import av
import numpy as np
import pytest

from pathlib import Path

from manim import DR, Circle, Create, Scene, Star, tempconfig
from manim.utils.commands import capture, get_video_metadata


class StarScene(Scene):
    def construct(self):
        circle = Circle(fill_opacity=1, color="#ff0000")
        circle.to_corner(DR).shift(DR)
        self.add(circle)
        star = Star()
        self.play(Create(star))
        click_path = Path(__file__).parent.parent.parent / "docs" / "source" / "_static" / "click.wav"
        self.add_sound(click_path)
        self.wait()


@pytest.mark.slow
@pytest.mark.parametrize(
    "format, transparent, codec, pixel_format",
    [
        ("mp4", False, "h264", "yuv420p"),
        ("mov", False, "h264", "yuv420p"),
        ("mov", True, "qtrle", "argb"),
        ("webm", False, "vp9", "yuv420p"),
        ("webm", True, "vp9", "yuv420p"),
        ("gif", False, "gif", "bgra"),
        ("gif", True, "gif", "bgra"),
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
    target_metadata = {
        "width": 854,
        "height": 480,
        "nb_frames": "30",
        "duration": "2.000000",
        "avg_frame_rate": "15/1",
        "codec_name": codec,
        "pix_fmt": pixel_format,
    }
    if format == "gif":  # reported duration + avg_frame_rate is slightly off for gifs
        del metadata["duration"], metadata["avg_frame_rate"]
        del target_metadata["duration"], target_metadata["avg_frame_rate"]
    assert metadata == target_metadata

    with av.open(video_path) as container:
        if transparent and format == "webm":
            from av.codec.context import CodecContext

            context = CodecContext.create("libvpx-vp9", "r")
            packet = next(container.demux(video=0))
            first_frame = context.decode(packet)[0].to_ndarray(format="argb")
        else:
            first_frame = next(container.decode(video=0))
            if format == "gif" and transparent:
                first_frame = first_frame.to_ndarray(format="argb")
            elif format == "gif" and not transparent:
                first_frame = first_frame.to_ndarray(format="rgb24")
            else:
                first_frame = first_frame.to_ndarray()

        target_rgba_corner = (
            np.array([0, 0, 0, 0]) if transparent else np.array(16, dtype=np.uint8)
        )
        if format == "gif":
            target_rgba_corner = (
                np.array([0, 255, 255, 255], dtype=np.uint8)
                if transparent
                else np.array([0, 0, 0], dtype=np.uint8)
            )
        np.testing.assert_array_equal(first_frame[0, 0], target_rgba_corner)

        target_rgba_center = (
            np.array([255, 255, 0, 0])
            if transparent  # components (A, R, G, B)
            else np.array(240, dtype=np.uint8)
        )
        if format == "gif" and not transparent:
            target_rgba_center = np.array([255, 0, 0], dtype=np.uint8)
        np.testing.assert_allclose(first_frame[-1, -1], target_rgba_center, atol=5)


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
