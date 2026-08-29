import sys
from fractions import Fraction
from pathlib import Path
from unittest.mock import Mock

import av
import numpy as np
import pytest

from manim import DR, Circle, Create, Scene, Star, tempconfig
from manim._config.output import OutputFormat, OutputSpec
from manim.scene.scene_file_writer import SceneFileWriter, to_av_frame_rate
from manim.utils.commands import capture, get_video_metadata


class StarScene(Scene):
    def construct(self):
        circle = Circle(fill_opacity=1, color="#ff0000")
        circle.to_corner(DR).shift(DR)
        self.add(circle)
        star = Star()
        self.play(Create(star))
        click_path = (
            Path(__file__).parent.parent.parent
            / "docs"
            / "source"
            / "_static"
            / "click.wav"
        )
        self.add_sound(click_path)
        self.wait()


@pytest.mark.slow
@pytest.mark.parametrize(
    "transparent",
    [False, True],
)
def test_gif_writing(config, tmp_path, transparent):
    output_filename = f"gif_{'transparent' if transparent else 'opaque'}"
    with tempconfig(
        {
            "media_dir": tmp_path,
            "quality": "low_quality",
            "format": "gif",
            "transparent": transparent,
            "output_file": output_filename,
        }
    ):
        StarScene().render()

    video_path = tmp_path / "videos" / "480p15" / f"{output_filename}.gif"
    assert video_path.exists()
    metadata = get_video_metadata(video_path)
    # reported duration + avg_frame_rate is slightly off for gifs
    del metadata["duration"], metadata["avg_frame_rate"]
    target_metadata = {
        "width": 854,
        "height": 480,
        "nb_frames": "30",
        "codec_name": "gif",
        "pix_fmt": "bgra",
    }
    assert metadata == target_metadata

    with av.open(video_path) as container:
        first_frame = next(container.decode(video=0))
        frame_format = "argb" if transparent else "rgb24"
        first_frame = first_frame.to_ndarray(format=frame_format)

    target_rgba_corner = (
        np.array([0, 255, 255, 255], dtype=np.uint8)
        if transparent
        else np.array([0, 0, 0], dtype=np.uint8)
    )
    np.testing.assert_array_equal(first_frame[0, 0], target_rgba_corner)

    target_rgba_center = (
        np.array([255, 255, 0, 0])  # components (A, R, G, B)
        if transparent
        else np.array([255, 0, 0], dtype=np.uint8)
    )
    np.testing.assert_allclose(first_frame[-1, -1], target_rgba_center, atol=5)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("format", "transparent", "codec", "pixel_format"),
    [
        ("mp4", False, "h264", "yuv420p"),
        ("mov", False, "h264", "yuv420p"),
        ("mov", True, "qtrle", "argb"),
        ("webm", False, "vp9", "yuv420p"),
        ("webm", True, "vp9", "yuv420p"),
    ],
)
def test_codecs(config, tmp_path, format, transparent, codec, pixel_format):
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
    assert metadata == target_metadata

    with av.open(video_path) as container:
        if transparent and format == "webm":
            from av.codec.context import CodecContext

            context = CodecContext.create("libvpx-vp9", "r")
            packet = next(container.demux(video=0))
            first_frame = context.decode(packet)[0].to_ndarray(format="argb")
        else:
            first_frame = next(container.decode(video=0)).to_ndarray()

        has_samples = [
            np.any(frame.to_ndarray()) for frame in container.decode(audio=0)
        ]
        assert any(has_samples), "All audio samples are zero, this is not intended"

    target_rgba_corner = (
        np.array([0, 0, 0, 0]) if transparent else np.array(16, dtype=np.uint8)
    )
    np.testing.assert_array_equal(first_frame[0, 0], target_rgba_corner)

    target_rgba_center = (
        np.array([255, 255, 0, 0])  # components (A, R, G, B)
        if transparent
        else np.array(240, dtype=np.uint8)
    )
    np.testing.assert_allclose(first_frame[-1, -1], target_rgba_center, atol=5)


def test_scene_with_non_raw_or_wav_audio(config, manim_caplog):
    class SceneWithMP3(Scene):
        def construct(self):
            file_path = Path(__file__).parent / "click.mp3"
            self.add_sound(file_path)
            self.wait()

    SceneWithMP3().render()
    assert "click.mp3 to .wav" in manim_caplog.text


@pytest.mark.slow
def test_unicode_partial_movie(config, tmpdir, simple_scenes_path):
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


def test_frame_rates():
    assert to_av_frame_rate(25) == Fraction(25, 1)
    assert to_av_frame_rate(24.0) == Fraction(24, 1)
    assert to_av_frame_rate(23.976) == Fraction(24 * 1000, 1001)
    assert to_av_frame_rate(23.98) == Fraction(24 * 1000, 1001)
    assert to_av_frame_rate(59.94) == Fraction(60 * 1000, 1001)


def _new_file_writer(scene_name: str) -> SceneFileWriter:
    renderer = Mock()
    renderer.num_plays = 0
    return SceneFileWriter(
        renderer,
        scene_name,
        OutputSpec(
            OutputFormat.MP4,
            transparent=False,
            save_sections=False,
            fallback_to_still=False,
        ),
    )


def test_clean_cache_ignores_hidden_files(config, tmp_path):
    # macOS leaves resource forks (._*.mp4) and .DS_Store files in the
    # partial movie directory; they must not be counted against
    # max_files_cached nor be deleted, see issue #3234.
    with tempconfig({"media_dir": tmp_path, "format": "mp4"}):
        writer = _new_file_writer("CacheCleaningScene")
        cache_dir = writer.partial_movie_directory

        for name in ["00001.mp4", ".DS_Store", "._00001.mp4"]:
            (cache_dir / name).touch()

        config.max_files_cached = 1
        # The hidden files must not count towards the limit: with a single
        # real partial movie file cached, nothing must be evicted.
        writer.clean_cache()

        assert (cache_dir / "00001.mp4").exists()
        assert (cache_dir / ".DS_Store").exists()
        assert (cache_dir / "._00001.mp4").exists()

        config.max_files_cached = 0
        writer.clean_cache()

        assert not (cache_dir / "00001.mp4").exists()
        assert (cache_dir / ".DS_Store").exists()
        assert (cache_dir / "._00001.mp4").exists()


def test_flush_cache_directory_ignores_hidden_files(config, tmp_path):
    with tempconfig({"media_dir": tmp_path, "format": "mp4"}):
        writer = _new_file_writer("CacheFlushingScene")
        cache_dir = writer.partial_movie_directory

        for name in ["00001.mp4", "00002.mp4", ".DS_Store", "._00001.mp4"]:
            (cache_dir / name).touch()
        (cache_dir / "partial_movie_file_list.txt").touch()

        writer.flush_cache_directory()

        assert not (cache_dir / "00001.mp4").exists()
        assert not (cache_dir / "00002.mp4").exists()
        assert (cache_dir / "partial_movie_file_list.txt").exists()
        assert (cache_dir / ".DS_Store").exists()
        assert (cache_dir / "._00001.mp4").exists()


def test_clean_cache_tolerates_vanishing_files(config, tmp_path, monkeypatch):
    # A file can disappear between listing the directory and unlinking it
    # (e.g. Finder removing a transient resource fork); clean_cache must
    # not raise FileNotFoundError in that case.
    with tempconfig({"media_dir": tmp_path, "format": "mp4"}):
        writer = _new_file_writer("VanishingFileScene")
        cache_dir = writer.partial_movie_directory

        survivor = cache_dir / "00001.mp4"
        survivor.touch()
        ghost = cache_dir / "00002.mp4"
        monkeypatch.setattr(
            writer, "_cached_partial_movie_files", lambda: [survivor, ghost]
        )

        config.max_files_cached = 0
        writer.clean_cache()

        assert not survivor.exists()


def test_clean_cache_does_not_evict_for_vanished_file(config, tmp_path, monkeypatch):
    with tempconfig({"media_dir": tmp_path, "format": "mp4"}):
        writer = _new_file_writer("VanishedFileEvictionScene")
        cache_dir = writer.partial_movie_directory

        survivors = [cache_dir / f"{index:05}.mp4" for index in range(2)]
        for survivor in survivors:
            survivor.touch()
        ghost = cache_dir / "00002.mp4"
        monkeypatch.setattr(
            writer,
            "_cached_partial_movie_files",
            lambda: [*survivors, ghost],
        )

        config.max_files_cached = len(survivors)
        writer.clean_cache()

        assert all(survivor.exists() for survivor in survivors)
