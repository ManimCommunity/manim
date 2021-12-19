import os
import tempfile
from pathlib import Path

import numpy as np

from manim import WHITE, Scene, Square, Tex, Text, config, tempconfig
from tests.assert_utils import assert_dir_exists, assert_dir_filled, assert_file_exists


def test_tempconfig():
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
            assert np.allclose(config[k], v)
        else:
            assert config[k] == v


class MyScene(Scene):
    def construct(self):
        self.add(Square())
        self.add(Text("Prepare for unforeseen consequencesλ"))
        self.add(Tex(r"$\lambda$"))
        self.wait(1)


def test_transparent():
    """Test the 'transparent' config option."""
    orig_verbosity = config["verbosity"]
    config["verbosity"] = "ERROR"

    with tempconfig({"dry_run": True}):
        scene = MyScene()
        scene.render()
        frame = scene.renderer.get_frame()
    assert np.allclose(frame[0, 0], [0, 0, 0, 255])

    with tempconfig({"transparent": True, "dry_run": True}):
        scene = MyScene()
        scene.render()
        frame = scene.renderer.get_frame()
        assert np.allclose(frame[0, 0], [0, 0, 0, 0])

    config["verbosity"] = orig_verbosity


def test_background_color():
    """Test the 'background_color' config option."""
    with tempconfig({"background_color": WHITE, "verbosity": "ERROR", "dry_run": True}):
        scene = MyScene()
        scene.render()
        frame = scene.renderer.get_frame()
        assert np.allclose(frame[0, 0], [255, 255, 255, 255])


def test_digest_file(tmp_path):
    """Test that a config file can be digested programmatically."""
    with tempconfig({}):
        tmp_cfg = tempfile.NamedTemporaryFile("w", dir=tmp_path, delete=False)
        tmp_cfg.write(
            """
            [CLI]
            media_dir = this_is_my_favorite_path
            video_dir = {media_dir}/videos
            sections_dir = {media_dir}/{scene_name}/prepare_for_unforeseen_consequences
            frame_height = 10
            """,
        )
        tmp_cfg.close()
        config.digest_file(tmp_cfg.name)

        assert config.get_dir("media_dir") == Path("this_is_my_favorite_path")
        assert config.get_dir("video_dir") == Path("this_is_my_favorite_path/videos")
        assert config.get_dir("sections_dir", scene_name="test") == Path(
            "this_is_my_favorite_path/test/prepare_for_unforeseen_consequences"
        )


def test_custom_dirs(tmp_path):
    with tempconfig(
        {
            "media_dir": tmp_path,
            "save_sections": True,
            "frame_rate": 15,
            "pixel_height": 854,
            "pixel_width": 480,
            "save_sections": True,
            "sections_dir": "{media_dir}/test_sections",
            "video_dir": "{media_dir}/test_video",
            "partial_movie_dir": "{media_dir}/test_partial_movie_dir",
            "images_dir": "{media_dir}/test_images",
            "text_dir": "{media_dir}/test_text",
            "tex_dir": "{media_dir}/test_tex",
        }
    ):
        scene = MyScene()
        scene.render()

        assert_dir_filled(os.path.join(tmp_path, "test_sections"))
        assert_file_exists(os.path.join(tmp_path, "test_sections", "MyScene.json"))

        assert_dir_filled(os.path.join(tmp_path, "test_video"))
        assert_file_exists(os.path.join(tmp_path, "test_video", "MyScene.mp4"))

        assert_dir_filled(os.path.join(tmp_path, "test_partial_movie_dir"))
        assert_file_exists(
            os.path.join(
                tmp_path, "test_partial_movie_dir", "partial_movie_file_list.txt"
            )
        )

        # TODO: another example with image output would be nice
        assert_dir_exists(os.path.join(tmp_path, "test_images"))

        assert_dir_filled(os.path.join(tmp_path, "test_text"))
        assert_dir_filled(os.path.join(tmp_path, "test_tex"))
        # TODO: testing the log dir would be nice but it doesn't get generated for some reason and test crashes when setting "log_to_file" to True


def test_frame_size(tmp_path):
    """Test that the frame size can be set via config file."""
    assert np.allclose(config.aspect_ratio, config.pixel_width / config.pixel_height)
    assert np.allclose(config.frame_height, 8.0)

    with tempconfig({}):
        tmp_cfg = tempfile.NamedTemporaryFile("w", dir=tmp_path, delete=False)
        tmp_cfg.write(
            """
            [CLI]
            pixel_height = 10
            pixel_width = 10
            """,
        )
        tmp_cfg.close()
        config.digest_file(tmp_cfg.name)

        # aspect ratio is set using pixel measurements
        assert np.allclose(config.aspect_ratio, 1.0)
        # if not specified in the cfg file, frame_width is set using the aspect ratio
        assert np.allclose(config.frame_height, 8.0)
        assert np.allclose(config.frame_width, 8.0)

    with tempconfig({}):
        tmp_cfg = tempfile.NamedTemporaryFile("w", dir=tmp_path, delete=False)
        tmp_cfg.write(
            """
            [CLI]
            pixel_height = 10
            pixel_width = 10
            frame_height = 10
            frame_width = 10
            """,
        )
        tmp_cfg.close()
        config.digest_file(tmp_cfg.name)

        assert np.allclose(config.aspect_ratio, 1.0)
        # if both are specified in the cfg file, the aspect ratio is ignored
        assert np.allclose(config.frame_height, 10.0)
        assert np.allclose(config.frame_width, 10.0)


def test_temporary_dry_run():
    """Test that tempconfig correctly restores after setting dry_run."""
    assert config["write_to_movie"]
    assert not config["save_last_frame"]

    with tempconfig({"dry_run": True}):
        assert not config["write_to_movie"]
        assert not config["save_last_frame"]

    assert config["write_to_movie"]
    assert not config["save_last_frame"]


def test_dry_run_with_png_format():
    """Test that there are no exceptions when running a png without output"""
    with tempconfig(
        {"dry_run": True, "write_to_movie": False, "disable_caching": True}
    ):
        assert config["dry_run"] is True
        scene = MyScene()
        scene.render()


def test_dry_run_with_png_format_skipped_animations():
    """Test that there are no exceptions when running a png without output and skipped animations"""
    with tempconfig(
        {"dry_run": True, "write_to_movie": False, "disable_caching": True}
    ):
        assert config["dry_run"] is True
        scene = MyScene(skip_animations=True)
        scene.render()
