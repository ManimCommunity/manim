from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import av
import pytest

from manim import Scene, capture, tempconfig
from tests.assert_utils import assert_dir_exists, assert_dir_not_exists

from ..utils.video_tester import video_comparison


@pytest.mark.slow
@video_comparison(
    "SceneWithDisabledSections.json",
    "videos/simple_scenes/480p15/SquareToCircle.mp4",
)
def test_no_sections(tmp_path, manim_cfg_file, simple_scenes_path):
    scene_name = "SquareToCircle"
    command = [
        sys.executable,
        "-m",
        "manim",
        "-ql",
        "--media_dir",
        str(tmp_path),
        str(simple_scenes_path),
        scene_name,
    ]
    _, err, exit_code = capture(command)
    assert exit_code == 0, err

    scene_dir = tmp_path / "videos" / "simple_scenes" / "480p15"
    assert_dir_exists(scene_dir)
    assert_dir_not_exists(scene_dir / "sections")


@pytest.mark.slow
@video_comparison(
    "SceneWithEnabledSections.json",
    "videos/simple_scenes/480p15/SquareToCircle.mp4",
)
def test_sections(tmp_path, manim_cfg_file, simple_scenes_path):
    scene_name = "SquareToCircle"
    command = [
        sys.executable,
        "-m",
        "manim",
        "-ql",
        "--save_sections",
        "--media_dir",
        str(tmp_path),
        str(simple_scenes_path),
        scene_name,
    ]
    _, err, exit_code = capture(command)
    assert exit_code == 0, err

    scene_dir = tmp_path / "videos" / "simple_scenes" / "480p15"
    assert_dir_exists(scene_dir)
    assert_dir_exists(scene_dir / "sections")


@pytest.mark.slow
@video_comparison(
    "SceneWithSections.json",
    "videos/simple_scenes/480p15/SceneWithSections.mp4",
)
def test_many_sections(tmp_path, manim_cfg_file, simple_scenes_path):
    scene_name = "SceneWithSections"
    command = [
        sys.executable,
        "-m",
        "manim",
        "-ql",
        "--save_sections",
        "--media_dir",
        str(tmp_path),
        str(simple_scenes_path),
        scene_name,
    ]
    _, err, exit_code = capture(command)
    assert exit_code == 0, err


@pytest.mark.slow
@video_comparison(
    "SceneWithSkipAnimations.json",
    "videos/simple_scenes/480p15/ElaborateSceneWithSections.mp4",
)
def test_skip_animations(tmp_path, manim_cfg_file, simple_scenes_path):
    scene_name = "ElaborateSceneWithSections"
    command = [
        sys.executable,
        "-m",
        "manim",
        "-ql",
        "--save_sections",
        "--media_dir",
        str(tmp_path),
        str(simple_scenes_path),
        scene_name,
    ]
    _, err, exit_code = capture(command)
    assert exit_code == 0, err


@pytest.mark.parametrize("with_sound", [False, True])
@pytest.mark.parametrize("save_sections", [False, True])
def test_mp4_section_chapters(config, tmp_path, with_sound, save_sections):
    class ChapterScene(Scene):
        def construct(self):
            self.next_section("开场：Introduction")
            if with_sound:
                self.add_sound(
                    Path(__file__).parents[2] / "docs/source/_static/click.wav"
                )
            self.wait(0.11)  # Static waits truncate to one encoded frame at 15 fps.
            self.wait(0.21)  # Three frames, sharing the first chapter.
            self.next_section("empty")
            self.next_section("skipped", skip_animations=True)
            self.wait(1)
            self.next_section("Conclusion")
            self.wait(0.14)  # Two frames.
            self.next_section("trailing empty")

    with tempconfig(
        {
            "media_dir": tmp_path,
            "pixel_width": 64,
            "pixel_height": 64,
            "frame_rate": 15,
            "format": "mp4",
            "save_sections": save_sections,
        }
    ):
        # Cached segments must produce the same chapter timeline on a second render.
        for _ in range(2):
            scene = ChapterScene()
            scene.render()
            with av.open(scene.renderer.file_writer.movie_file_path) as video:
                chapters = video.chapters()
                assert bool(video.streams.audio) == with_sound
                assert len(list(video.decode(video=0))) == 6
            if not save_sections:
                assert chapters == []
                continue
            assert [chapter["metadata"]["title"] for chapter in chapters] == [
                "开场：Introduction",
                "Conclusion",
            ]
            # MP4 chapter timestamps are quantized to milliseconds by the muxer.
            boundaries = [
                (
                    chapter["start"] * chapter["time_base"],
                    chapter["end"] * chapter["time_base"],
                )
                for chapter in chapters
            ]
            assert boundaries == [
                (0, Fraction(267, 1000)),
                (Fraction(267, 1000), Fraction(2, 5)),
            ]
