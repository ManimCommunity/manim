import sys
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from manim.__main__ import main
from manim.utils.file_ops import add_version_before_extension

from ..utils.video_tester import *


@pytest.mark.slow
@video_comparison(
    "SquareToCircleWithDefaultValues.json",
    "videos/simple_scenes/1080p60/SquareToCircle.mp4",
)
def test_basic_scene_with_default_values(tmp_path, manim_cfg_file, simple_scenes_path):
    scene_name = "SquareToCircle"
    command = [
        sys.executable,
        "-m",
        "manim",
        "--media_dir",
        str(tmp_path),
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err


@pytest.mark.slow
@video_comparison(
    "SquareToCircleWithlFlag.json", "videos/simple_scenes/480p15/SquareToCircle.mp4"
)
def test_basic_scene_l_flag(tmp_path, manim_cfg_file, simple_scenes_path):
    scene_name = "SquareToCircle"
    command = [
        sys.executable,
        "-m",
        "manim",
        "-ql",
        "--media_dir",
        str(tmp_path),
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err


@pytest.mark.slow
@video_comparison(
    "SceneWithMultipleCallsWithNFlag.json",
    "videos/simple_scenes/1080p60/SceneWithMultipleCalls.mp4",
)
def test_n_flag(tmp_path, simple_scenes_path):
    scene_name = "SceneWithMultipleCalls"
    command = [
        sys.executable,
        "-m",
        "manim",
        "-n 3,6",
        "--media_dir",
        str(tmp_path),
        simple_scenes_path,
        scene_name,
    ]
    _, err, exit_code = capture(command)
    assert exit_code == 0, err


@pytest.mark.slow
def test_s_flag_no_animations(tmp_path, manim_cfg_file, simple_scenes_path):
    scene_name = "NoAnimations"
    command = [
        sys.executable,
        "-m",
        "manim",
        "-ql",
        "-s",
        "--media_dir",
        str(tmp_path),
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err

    exists = (tmp_path / "videos").exists()
    assert not exists, "running manim with -s flag rendered a video"

    is_empty = not any((tmp_path / "images" / "simple_scenes").iterdir())
    assert not is_empty, "running manim with -s flag did not render an image"


@pytest.mark.slow
def test_s_flag(tmp_path, manim_cfg_file, simple_scenes_path):
    scene_name = "SquareToCircle"
    command = [
        sys.executable,
        "-m",
        "manim",
        "-ql",
        "-s",
        "--media_dir",
        str(tmp_path),
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err

    exists = (tmp_path / "videos").exists()
    assert not exists, "running manim with -s flag rendered a video"

    is_empty = not any((tmp_path / "images" / "simple_scenes").iterdir())
    assert not is_empty, "running manim with -s flag did not render an image"


@pytest.mark.slow
def test_r_flag(tmp_path, manim_cfg_file, simple_scenes_path):
    scene_name = "SquareToCircle"
    command = [
        sys.executable,
        "-m",
        "manim",
        "-ql",
        "-s",
        "--media_dir",
        str(tmp_path),
        "-r",
        "200,100",
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err

    is_not_empty = any((tmp_path / "images").iterdir())
    assert is_not_empty, "running manim with -s, -r flag did not render a file"

    filename = add_version_before_extension(
        tmp_path / "images" / "simple_scenes" / "SquareToCircle.png"
    )
    assert np.asarray(Image.open(filename)).shape == (100, 200, 4)


@pytest.mark.slow
def test_a_flag(tmp_path, manim_cfg_file, infallible_scenes_path):
    command = [
        sys.executable,
        "-m",
        "manim",
        "-ql",
        "--media_dir",
        str(tmp_path),
        "-a",
        infallible_scenes_path,
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err

    one_is_not_empty = (
        tmp_path / "videos" / "infallible_scenes" / "480p15" / "Wait1.mp4"
    ).is_file()
    assert one_is_not_empty, "running manim with -a flag did not render the first scene"

    two_is_not_empty = (
        tmp_path / "videos" / "infallible_scenes" / "480p15" / "Wait2.mp4"
    ).is_file()
    assert (
        two_is_not_empty
    ), "running manim with -a flag did not render the second scene"


@pytest.mark.slow
def test_custom_folders(tmp_path, manim_cfg_file, simple_scenes_path):
    scene_name = "SquareToCircle"
    command = [
        sys.executable,
        "-m",
        "manim",
        "-ql",
        "-s",
        "--media_dir",
        str(tmp_path),
        "--custom_folders",
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err

    exists = (tmp_path / "videos").exists()
    assert not exists, "--custom_folders produced a 'videos/' dir"

    exists = add_version_before_extension(tmp_path / "SquareToCircle.png").exists()
    assert exists, "--custom_folders did not produce the output file"


@pytest.mark.slow
def test_dash_as_filename(tmp_path):
    code = (
        "class Test(Scene):\n"
        "    def construct(self):\n"
        "        self.add(Circle())\n"
        "        self.wait()"
    )
    command = [
        "-ql",
        "-s",
        "--media_dir",
        str(tmp_path),
        "-",
    ]
    runner = CliRunner()
    result = runner.invoke(main, command, input=code)
    assert result.exit_code == 0
    exists = add_version_before_extension(
        tmp_path / "images" / "-" / "Test.png"
    ).exists()
    assert exists, result.output
