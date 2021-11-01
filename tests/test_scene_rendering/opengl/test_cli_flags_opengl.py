import itertools
import os
import sys

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from manim import capture, get_video_metadata
from manim.__main__ import main
from manim.utils.file_ops import add_version_before_extension
from tests.utils.video_tester import video_comparison


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
        "--renderer",
        "opengl",
        "--write_to_movie",
        "--media_dir",
        str(tmp_path),
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err


@pytest.mark.slow()
def test_resolution_flag(tmp_path, manim_cfg_file, simple_scenes_path):
    scene_name = "NoAnimations"
    resolutions = [
        (720, 480),
        (1280, 720),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160),
        (640, 480),
        (800, 600),
    ]

    separators = [";", ",", "-"]

    for (width, height), separator in itertools.product(resolutions, separators):
        command = [
            sys.executable,
            "-m",
            "manim",
            "--media_dir",
            str(tmp_path),
            "--resolution",
            f"{width}{separator}{height}",
            str(simple_scenes_path),
            scene_name,
        ]

        _, err, exit_code = capture(command)
        assert exit_code == 0, err

        path = (
            tmp_path / "videos" / "simple_scenes" / f"{height}p60" / f"{scene_name}.mp4"
        )
        meta = get_video_metadata(path)
        assert (width, height) == (meta["width"], meta["height"])


@pytest.mark.slow
@video_comparison(
    "SquareToCircleWithlFlag.json",
    "videos/simple_scenes/480p15/SquareToCircle.mp4",
)
def test_basic_scene_l_flag(tmp_path, manim_cfg_file, simple_scenes_path):
    scene_name = "SquareToCircle"
    command = [
        sys.executable,
        "-m",
        "manim",
        "--renderer",
        "opengl",
        "-ql",
        "--write_to_movie",
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
        "--renderer",
        "opengl",
        "--write_to_movie",
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
        "--renderer",
        "opengl",
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
def test_image_output_for_static_scene(tmp_path, manim_cfg_file, simple_scenes_path):
    scene_name = "StaticScene"
    command = [
        sys.executable,
        "-m",
        "manim",
        "--renderer",
        "opengl",
        "-ql",
        "--media_dir",
        str(tmp_path),
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err

    exists = (tmp_path / "videos").exists()
    assert not exists, "running manim with static scene rendered a video"

    is_empty = not any((tmp_path / "images" / "simple_scenes").iterdir())
    assert not is_empty, "running manim without animations did not render an image"


@pytest.mark.slow
def test_no_image_output_with_interactive_embed(
    tmp_path, manim_cfg_file, simple_scenes_path
):
    """Check no image is output for a static scene when interactive embed is called"""
    scene_name = "InteractiveStaticScene"
    command = [
        sys.executable,
        "-m",
        "manim",
        "--renderer",
        "opengl",
        "-ql",
        "--media_dir",
        str(tmp_path),
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err

    exists = (tmp_path / "videos").exists()
    assert not exists, "running manim with static scene rendered a video"

    is_empty = len(os.listdir(tmp_path / "images" / "simple_scenes")) == 0
    assert (
        is_empty
    ), "running manim static scene with interactive embed rendered an image"


@pytest.mark.slow
def test_no_default_image_output_with_non_static_scene(
    tmp_path, manim_cfg_file, simple_scenes_path
):
    scene_name = "SceneWithNonStaticWait"
    command = [
        sys.executable,
        "-m",
        "manim",
        "--renderer",
        "opengl",
        "-ql",
        "--media_dir",
        str(tmp_path),
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err

    exists = (tmp_path / "videos").exists()
    assert not exists, "running manim with static scene rendered a video"

    is_empty = len(os.listdir(tmp_path / "images" / "simple_scenes")) == 0
    assert (
        is_empty
    ), "running manim static scene with interactive embed rendered an image"


@pytest.mark.slow
def test_image_output_for_static_scene_with_write_to_movie(
    tmp_path, manim_cfg_file, simple_scenes_path
):
    scene_name = "StaticScene"
    command = [
        sys.executable,
        "-m",
        "manim",
        "--write_to_movie",
        "--renderer",
        "opengl",
        "-ql",
        "--media_dir",
        str(tmp_path),
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err

    exists = len(os.listdir(tmp_path / "videos")) == 0
    assert not exists, "running manim with static scene rendered a video"

    is_empty = not any((tmp_path / "images" / "simple_scenes").iterdir())
    assert not is_empty, "running manim without animations did not render an image"


@pytest.mark.slow
def test_s_flag(tmp_path, manim_cfg_file, simple_scenes_path):
    scene_name = "SquareToCircle"
    command = [
        sys.executable,
        "-m",
        "manim",
        "--renderer",
        "opengl",
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
        "--renderer",
        "opengl",
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
        tmp_path / "images" / "simple_scenes" / "SquareToCircle.png",
    )
    assert np.asarray(Image.open(filename)).shape == (100, 200, 4)


@pytest.mark.slow
def test_a_flag(tmp_path, manim_cfg_file, infallible_scenes_path):
    command = [
        sys.executable,
        "-m",
        "manim",
        "--renderer",
        "opengl",
        "--write_to_movie",
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
        "--renderer",
        "opengl",
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
        "--renderer",
        "opengl",
        "-s",
        "--media_dir",
        str(tmp_path),
        "-",
    ]
    runner = CliRunner()
    result = runner.invoke(main, command, input=code)
    assert result.exit_code == 0
    exists = add_version_before_extension(
        tmp_path / "images" / "-" / "Test.png",
    ).exists()
    assert exists, result.output


@pytest.mark.slow
def test_gif_format_output(tmp_path, manim_cfg_file, simple_scenes_path):
    """Test only gif created with manim version in file name when --format gif is set"""
    scene_name = "SquareToCircle"
    command = [
        sys.executable,
        "-m",
        "manim",
        "--renderer",
        "opengl",
        "-ql",
        "--media_dir",
        str(tmp_path),
        "--format",
        "gif",
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err

    unexpected_mp4_path = (
        tmp_path / "videos" / "simple_scenes" / "480p15" / "SquareToCircle.mp4"
    )
    assert not unexpected_mp4_path.exists(), "unexpected mp4 file found at " + str(
        unexpected_mp4_path,
    )

    expected_gif_path = (
        tmp_path
        / "videos"
        / "simple_scenes"
        / "480p15"
        / add_version_before_extension("SquareToCircle.gif")
    )
    assert expected_gif_path.exists(), "gif file not found at " + str(expected_gif_path)


@pytest.mark.slow
def test_mp4_format_output(tmp_path, manim_cfg_file, simple_scenes_path):
    """Test only mp4 created without manim version in file name when --format mp4 is set"""
    scene_name = "SquareToCircle"
    command = [
        sys.executable,
        "-m",
        "manim",
        "--renderer",
        "opengl",
        "-ql",
        "--media_dir",
        str(tmp_path),
        "--format",
        "mp4",
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err

    unexpected_gif_path = (
        tmp_path
        / "videos"
        / "simple_scenes"
        / "480p15"
        / add_version_before_extension("SquareToCircle.gif")
    )
    assert not unexpected_gif_path.exists(), "unexpected gif file found at " + str(
        unexpected_gif_path,
    )

    expected_mp4_path = (
        tmp_path / "videos" / "simple_scenes" / "480p15" / "SquareToCircle.mp4"
    )
    assert expected_mp4_path.exists(), "expected mp4 file not found at " + str(
        expected_mp4_path,
    )


@pytest.mark.slow
def test_videos_not_created_when_png_format_set(
    tmp_path,
    manim_cfg_file,
    simple_scenes_path,
):
    """Test mp4 and gifs are not created when --format png is set"""
    scene_name = "SquareToCircle"
    command = [
        sys.executable,
        "-m",
        "manim",
        "--renderer",
        "opengl",
        "-ql",
        "--media_dir",
        str(tmp_path),
        "--format",
        "png",
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err

    unexpected_gif_path = (
        tmp_path
        / "videos"
        / "simple_scenes"
        / "480p15"
        / add_version_before_extension("SquareToCircle.gif")
    )
    assert not unexpected_gif_path.exists(), "unexpected gif file found at " + str(
        unexpected_gif_path,
    )

    unexpected_mp4_path = (
        tmp_path / "videos" / "simple_scenes" / "480p15" / "SquareToCircle.mp4"
    )
    assert not unexpected_mp4_path.exists(), "expected mp4 file not found at " + str(
        unexpected_mp4_path,
    )


@pytest.mark.slow
def test_images_are_created_when_png_format_set(
    tmp_path,
    manim_cfg_file,
    simple_scenes_path,
):
    """Test images are created in media directory when --format png is set"""
    scene_name = "SquareToCircle"
    command = [
        sys.executable,
        "-m",
        "manim",
        "--renderer",
        "opengl",
        "-ql",
        "--media_dir",
        str(tmp_path),
        "--format",
        "png",
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err

    expected_png_path = tmp_path / "images" / "simple_scenes" / "SquareToCircle0000.png"
    assert expected_png_path.exists(), "png file not found at " + str(expected_png_path)


@pytest.mark.slow
def test_images_are_zero_padded_when_zero_pad_set(
    tmp_path,
    manim_cfg_file,
    simple_scenes_path,
):
    """Test images are zero padded when --format png and --zero_pad n are set"""
    scene_name = "SquareToCircle"
    command = [
        sys.executable,
        "-m",
        "manim",
        "--renderer",
        "opengl",
        "-ql",
        "--media_dir",
        str(tmp_path),
        "--format",
        "png",
        "--zero_pad",
        "3",
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err

    unexpected_png_path = tmp_path / "images" / "simple_scenes" / "SquareToCircle0.png"
    assert not unexpected_png_path.exists(), "non zero padded png file found at " + str(
        unexpected_png_path,
    )

    expected_png_path = tmp_path / "images" / "simple_scenes" / "SquareToCircle000.png"
    assert expected_png_path.exists(), "png file not found at " + str(expected_png_path)


@pytest.mark.slow
def test_webm_format_output(tmp_path, manim_cfg_file, simple_scenes_path):
    """Test only webm created when --format webm is set"""
    scene_name = "SquareToCircle"
    command = [
        sys.executable,
        "-m",
        "manim",
        "--renderer",
        "opengl",
        "-ql",
        "--media_dir",
        str(tmp_path),
        "--format",
        "webm",
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err

    unexpected_mp4_path = (
        tmp_path / "videos" / "simple_scenes" / "480p15" / "SquareToCircle.mp4"
    )
    assert not unexpected_mp4_path.exists(), "unexpected mp4 file found at " + str(
        unexpected_mp4_path,
    )

    expected_webm_path = (
        tmp_path / "videos" / "simple_scenes" / "480p15" / "SquareToCircle.webm"
    )
    assert expected_webm_path.exists(), "expected webm file not found at " + str(
        expected_webm_path,
    )


@pytest.mark.slow
def test_default_format_output_for_transparent_flag(
    tmp_path,
    manim_cfg_file,
    simple_scenes_path,
):
    """Test .mov is created by default when transparent flag is set"""
    scene_name = "SquareToCircle"
    command = [
        sys.executable,
        "-m",
        "manim",
        "--renderer",
        "opengl",
        "-ql",
        "--write_to_movie",
        "--media_dir",
        str(tmp_path),
        "-t",
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err

    unexpected_webm_path = (
        tmp_path / "videos" / "simple_scenes" / "480p15" / "SquareToCircle.webm"
    )
    assert not unexpected_webm_path.exists(), "unexpected webm file found at " + str(
        unexpected_webm_path,
    )

    expected_mov_path = (
        tmp_path / "videos" / "simple_scenes" / "480p15" / "SquareToCircle.mov"
    )
    assert expected_mov_path.exists(), "expected .mov file not found at " + str(
        expected_mov_path,
    )


@pytest.mark.slow
def test_mov_can_be_set_as_output_format(tmp_path, manim_cfg_file, simple_scenes_path):
    """Test .mov is created by when set using --format mov arg"""
    scene_name = "SquareToCircle"
    command = [
        sys.executable,
        "-m",
        "manim",
        "--renderer",
        "opengl",
        "-ql",
        "--media_dir",
        str(tmp_path),
        "--format",
        "mov",
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err

    unexpected_webm_path = (
        tmp_path / "videos" / "simple_scenes" / "480p15" / "SquareToCircle.webm"
    )
    assert not unexpected_webm_path.exists(), "unexpected webm file found at " + str(
        unexpected_webm_path,
    )

    expected_mov_path = (
        tmp_path / "videos" / "simple_scenes" / "480p15" / "SquareToCircle.mov"
    )
    assert expected_mov_path.exists(), "expected .mov file not found at " + str(
        expected_mov_path,
    )
