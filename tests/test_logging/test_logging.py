import os
import re
from pathlib import Path

from manim import capture

from ..utils.logging_tester import *


@logs_comparison(
    "BasicSceneLoggingTest.txt",
    os.path.join("logs", "basic_scenes_square_to_circle_SquareToCircle.log"),
)
def test_logging_to_file(tmp_path, python_version):
    path_basic_scene = os.path.join(
        "tests",
        "test_logging",
        "basic_scenes_square_to_circle.py",
    )
    command = [
        python_version,
        "-m",
        "manim",
        "-ql",
        "-v",
        "DEBUG",
        "--log_to_file",
        "--media_dir",
        str(tmp_path),
        path_basic_scene,
        "SquareToCircle",
    ]
    _, err, exitcode = capture(command)
    assert exitcode == 0, err


@logs_comparison(
    "BasicSceneLoggingTest.txt",
    os.path.join("logs", "basic_scenes_square_to_circle.log"),
)
def test_logging_when_scene_is_not_specified(tmp_path, python_version):
    path_basic_scene = os.path.join(
        "tests",
        "test_logging",
        "basic_scenes_square_to_circle.py",
    )
    command = [
        python_version,
        "-m",
        "manim",
        "-ql",
        "-v",
        "DEBUG",
        "--log_to_file",
        "--media_dir",
        str(tmp_path),
        path_basic_scene,
    ]
    _, err, exitcode = capture(command)
    assert exitcode == 0, err


def test_error_logging(tmp_path, python_version):
    path_error_scene = Path("tests/test_logging/basic_scenes_error.py")

    command = [
        python_version,
        "-m",
        "manim",
        "-ql",
        "--media_dir",
        str(tmp_path),
        str(path_error_scene),
    ]

    _, err, exitcode = capture(command)
    assert exitcode != 0 and len(err) > 0
