from __future__ import annotations

import json
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


requires_chapters = pytest.mark.skipif(
    not hasattr(av.container.OutputContainer, "set_chapters"),
    reason="PyAV < 16 has no chapter API",
)


def _render_chapters(scene_class, tmp_path, **options):
    """Render ``scene_class`` and return its movie's chapters as ``(title, start, end)``."""
    with tempconfig(
        {
            "media_dir": tmp_path,
            "pixel_width": 64,
            "pixel_height": 64,
            "frame_rate": 15,
            "format": "mp4",
            **options,
        }
    ):
        scene = scene_class()
        scene.render()
        with av.open(scene.renderer.file_writer.movie_file_path) as video:
            return [
                (
                    chapter["metadata"]["title"],
                    chapter["start"] * chapter["time_base"],
                    chapter["end"] * chapter["time_base"],
                )
                for chapter in video.chapters()
            ]


@requires_chapters
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
            self.wait(1 / 15)  # One frame at 15 fps.
            self.wait(3 / 15)  # Three frames, sharing the first chapter.
            self.next_section("empty")
            self.next_section("skipped", skip_animations=True)
            self.wait(1)
            self.next_section("Conclusion")
            self.wait(2 / 15)  # Two frames.
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
            assert [chapter["metadata"]["title"] for chapter in chapters] == [
                "开场：Introduction",
                "Conclusion",
            ]
            # Chapter times are written in milliseconds.
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
            if save_sections:
                # Writing the chapters leaves the sections to the section videos,
                # which skip the empty trailing section like the chapters do.
                writer = scene.renderer.file_writer
                index = json.loads(writer.output_plan.section_index.read_text())
                assert [entry["name"] for entry in index] == [
                    "开场：Introduction",
                    "Conclusion",
                ]
                for entry in index:
                    assert (writer.sections_output_dir / entry["video"]).exists()


@requires_chapters
def test_chapter_of_autocreated_section_is_named_after_movie(config, tmp_path):
    class LateSections(Scene):
        def construct(self):
            self.wait(0.2)  # Three frames in the automatically created section.
            self.next_section("Named")
            self.wait(0.2)

    assert _render_chapters(LateSections, tmp_path) == [
        ("LateSections", 0, Fraction(1, 5)),
        ("Named", Fraction(1, 5), Fraction(2, 5)),
    ]


@requires_chapters
def test_scene_without_sections_has_no_chapters(config, tmp_path):
    class NoSections(Scene):
        def construct(self):
            self.wait(0.2)

    assert _render_chapters(NoSections, tmp_path) == []


@requires_chapters
def test_chapters_follow_selected_plays(config, tmp_path):
    class Selected(Scene):
        def construct(self):
            self.next_section("first")
            self.wait(0.2)
            self.next_section("second")
            self.wait(0.2)
            self.wait(0.4)

    # Only the last two plays are in the movie; the first section gets no chapter.
    assert _render_chapters(Selected, tmp_path, from_animation_number=1) == [
        ("second", 0, Fraction(3, 5)),
    ]
