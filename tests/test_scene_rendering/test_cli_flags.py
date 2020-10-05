import pytest
from manim import constants

from ..utils.video_tester import *

from manim.config.config_utils import _determine_quality, _parse_cli


@pytest.mark.slow
@video_comparison(
    "SquareToCircleWithDefaultValues.json",
    "videos/simple_scenes/1080p60/SquareToCircle.mp4",
)
def test_basic_scene_with_default_values(tmp_path, manim_cfg_file, simple_scenes_path):
    scene_name = "SquareToCircle"
    command = [
        "python",
        "-m",
        "manim",
        simple_scenes_path,
        scene_name,
        "--media_dir",
        str(tmp_path),
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err


@video_comparison(
    "SquareToCircleWithlFlag.json", "videos/simple_scenes/480p15/SquareToCircle.mp4"
)
def test_basic_scene_l_flag(tmp_path, manim_cfg_file, simple_scenes_path):
    scene_name = "SquareToCircle"
    command = [
        "python",
        "-m",
        "manim",
        simple_scenes_path,
        scene_name,
        "-ql",
        "--media_dir",
        str(tmp_path),
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
        "python",
        "-m",
        "manim",
        simple_scenes_path,
        scene_name,
        "-n 3,6",
        "--media_dir",
        str(tmp_path),
    ]
    _, err, exit_code = capture(command)
    assert exit_code == 0, err


def test_quality_flags():
    # Assert that quality is None when not specifying it
    parsed = _parse_cli([], False)

    assert not parsed.quality

    for quality in constants.QUALITIES.keys():
        # Assert that quality is properly set when using -q*
        arguments = f"-q{constants.QUALITIES[quality]}".split()
        parsed = _parse_cli(arguments, False)

        assert parsed.quality == constants.QUALITIES[quality]
        assert quality == _determine_quality(parsed)

        # Assert that quality is properly set when using -q *
        arguments = f"-q {constants.QUALITIES[quality]}".split()
        parsed = _parse_cli(arguments, False)

        assert parsed.quality == constants.QUALITIES[quality]
        assert quality == _determine_quality(parsed)

        # Assert that quality is properly set when using --quality *
        arguments = f"--quality {constants.QUALITIES[quality]}".split()
        parsed = _parse_cli(arguments, False)

        assert parsed.quality == constants.QUALITIES[quality]
        assert quality == _determine_quality(parsed)

        # Assert that quality is properly set when using -*_quality
        arguments = f"--{quality}".split()
        parsed = _parse_cli(arguments, False)

        assert getattr(parsed, quality)
        assert quality == _determine_quality(parsed)

        # Assert that *_quality is False when not specifying it
        parsed = _parse_cli([], False)

        assert not getattr(parsed, quality)
        assert "production" == _determine_quality(parsed)
