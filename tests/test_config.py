from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from manim import WHITE, Scene, Square, Tex, Text, tempconfig
from manim._config.utils import ManimConfig
from tests.assert_utils import assert_dir_exists, assert_dir_filled, assert_file_exists


def test_tempconfig(config):
    """Test the tempconfig context manager."""
    original = config.copy()

    with tempconfig({"frame_width": 100, "frame_height": 42}):
        # check that config was modified correctly
        assert config["frame_width"] == 100
        assert config["frame_height"] == 42

        # check that no keys are missing and no new keys were added
        assert set(original.keys()) == set(config.keys())

    # check that the keys are still untouched
    assert set(original.keys()) == set(config.keys())

    # check that config is correctly restored
    for k, v in original.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_allclose(config[k], v)
        else:
            assert config[k] == v


@pytest.mark.parametrize(
    ("format", "expected_file_extension"),
    [
        ("mp4", ".mp4"),
        ("webm", ".webm"),
        ("mov", ".mov"),
        ("gif", ".mp4"),
    ],
)
def test_resolve_file_extensions(config, format, expected_file_extension):
    config.format = format
    assert config.movie_file_extension == expected_file_extension


class MyScene(Scene):
    def construct(self):
        self.add(Square())
        self.add(Text("Prepare for unforeseen consequencesλ"))
        self.add(Tex(r"$\lambda$"))
        self.wait(1)


def test_transparent(config):
    """Test the 'transparent' config option."""
    config.verbosity = "ERROR"
    config.dry_run = True

    scene = MyScene()
    scene.render()
    frame = scene.renderer.get_frame()
    np.testing.assert_allclose(frame[0, 0], [0, 0, 0, 255])

    config.transparent = True

    scene = MyScene()
    scene.render()
    frame = scene.renderer.get_frame()
    np.testing.assert_allclose(frame[0, 0], [0, 0, 0, 0])


def test_transparent_by_background_opacity(config, dry_run):
    config.background_opacity = 0.5
    assert config.transparent is True

    scene = MyScene()
    scene.render()
    frame = scene.renderer.get_frame()
    np.testing.assert_allclose(frame[0, 0], [0, 0, 0, 127])
    assert config.movie_file_extension == ".mov"
    assert config.transparent is True


def test_background_color(config):
    """Test the 'background_color' config option."""
    config.background_color = WHITE
    config.verbosity = "ERROR"
    config.dry_run = True

    scene = MyScene()
    scene.render()
    frame = scene.renderer.get_frame()
    np.testing.assert_allclose(frame[0, 0], [255, 255, 255, 255])


def test_digest_file(tmp_path, config):
    """Test that a config file can be digested programmatically."""
    with tempfile.NamedTemporaryFile("w", dir=tmp_path, delete=False) as tmp_cfg:
        tmp_cfg.write(
            """
            [CLI]
            media_dir = this_is_my_favorite_path
            video_dir = {media_dir}/videos
            sections_dir = {media_dir}/{scene_name}/prepare_for_unforeseen_consequences
            frame_height = 10
            """,
        )
    config.digest_file(tmp_cfg.name)

    assert config.get_dir("media_dir") == Path("this_is_my_favorite_path")
    assert config.get_dir("video_dir") == Path("this_is_my_favorite_path/videos")
    assert config.get_dir("sections_dir", scene_name="test") == Path(
        "this_is_my_favorite_path/test/prepare_for_unforeseen_consequences"
    )


def test_custom_dirs(tmp_path, config):
    config.media_dir = tmp_path
    config.save_sections = True
    config.log_to_file = True
    config.frame_rate = 15
    config.pixel_height = 854
    config.pixel_width = 480
    config.sections_dir = "{media_dir}/test_sections"
    config.video_dir = "{media_dir}/test_video"
    config.partial_movie_dir = "{media_dir}/test_partial_movie_dir"
    config.images_dir = "{media_dir}/test_images"
    config.text_dir = "{media_dir}/test_text"
    config.tex_dir = "{media_dir}/test_tex"
    config.log_dir = "{media_dir}/test_log"

    scene = MyScene()
    scene.render()
    tmp_path = Path(tmp_path)
    assert_dir_filled(tmp_path / "test_sections")
    assert_file_exists(tmp_path / "test_sections/MyScene.json")

    assert_dir_filled(tmp_path / "test_video")
    assert_file_exists(tmp_path / "test_video/MyScene.mp4")

    assert_dir_filled(tmp_path / "test_partial_movie_dir")
    assert_file_exists(tmp_path / "test_partial_movie_dir/partial_movie_file_list.txt")

    # TODO: another example with image output would be nice
    assert_dir_exists(tmp_path / "test_images")

    assert_dir_filled(tmp_path / "test_text")
    assert_dir_filled(tmp_path / "test_tex")
    assert_dir_filled(tmp_path / "test_log")


def test_pixel_dimensions(tmp_path, config):
    with tempfile.NamedTemporaryFile("w", dir=tmp_path, delete=False) as tmp_cfg:
        tmp_cfg.write(
            """
            [CLI]
            pixel_height = 10
            pixel_width = 10
            """,
        )
    config.digest_file(tmp_cfg.name)

    # aspect ratio is set using pixel measurements
    np.testing.assert_allclose(config.aspect_ratio, 1.0)
    # if not specified in the cfg file, frame_width is set using the aspect ratio
    np.testing.assert_allclose(config.frame_height, 8.0)
    np.testing.assert_allclose(config.frame_width, 8.0)


def test_frame_size(tmp_path, config):
    """Test that the frame size can be set via config file."""
    np.testing.assert_allclose(
        config.aspect_ratio, config.pixel_width / config.pixel_height
    )
    np.testing.assert_allclose(config.frame_height, 8.0)

    with tempfile.NamedTemporaryFile("w", dir=tmp_path, delete=False) as tmp_cfg:
        tmp_cfg.write(
            """
            [CLI]
            pixel_height = 10
            pixel_width = 10
            frame_height = 10
            frame_width = 10
            """,
        )
    config.digest_file(tmp_cfg.name)

    np.testing.assert_allclose(config.aspect_ratio, 1.0)
    # if both are specified in the cfg file, the aspect ratio is ignored
    np.testing.assert_allclose(config.frame_height, 10.0)
    np.testing.assert_allclose(config.frame_width, 10.0)


def test_temporary_dry_run(config):
    """Test that tempconfig correctly restores after setting dry_run."""
    assert config["write_to_movie"]
    assert not config["save_last_frame"]

    with tempconfig({"dry_run": True}):
        assert not config["write_to_movie"]
        assert not config["save_last_frame"]

    assert config["write_to_movie"]
    assert not config["save_last_frame"]


def test_dry_run_with_png_format(config, dry_run):
    """Test that there are no exceptions when running a png without output"""
    config.write_to_movie = False
    config.disable_caching = True
    assert config.dry_run is True
    scene = MyScene()
    scene.render()


def test_dry_run_with_png_format_skipped_animations(config, dry_run):
    """Test that there are no exceptions when running a png without output and skipped animations"""
    config.write_to_movie = False
    config.disable_caching = True
    assert config["dry_run"] is True
    scene = MyScene(skip_animations=True)
    scene.render()


def test_tex_template_file(tmp_path):
    """Test that a custom tex template file can be set from a config file."""
    tex_file = Path(tmp_path / "my_template.tex")
    tex_file.write_text("Hello World!")
    with tempfile.NamedTemporaryFile("w", dir=tmp_path, delete=False) as tmp_cfg:
        tmp_cfg.write(
            f"""
            [CLI]
            tex_template_file = {tex_file}
            """,
        )

    custom_config = ManimConfig().digest_file(tmp_cfg.name)

    assert Path(custom_config.tex_template_file) == tex_file
    assert custom_config.tex_template.body == "Hello World!"


def test_from_to_animations_only_first_animation(config):
    config: ManimConfig
    config.from_animation_number = 0
    config.upto_animation_number = 0

    class SceneWithTwoAnimations(Scene):
        def construct(self):
            self.after_first_animation = False
            s = Square()
            self.add(s)
            self.play(s.animate.scale(2))
            self.renderer.update_skipping_status()
            self.after_first_animation = True
            self.play(s.animate.scale(2))

    scene = SceneWithTwoAnimations()
    scene.render()

    assert scene.after_first_animation is False
