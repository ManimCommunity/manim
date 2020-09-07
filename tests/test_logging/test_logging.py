import subprocess
import os
import sys
import pytest
import re

from ..utils.commands import capture
from ..utils.logging_tester import *


@logs_comparison(
    "BasicSceneLoggingTest.txt", os.path.join("logs", "SquareToCircle.log")
)
def test_logging_to_file(tmp_path, python_version):
    path_basic_scene = os.path.join("tests", "test_logging", "basic_scenes.py")
    # os.makedirs(path_output, exist_ok=True)
    command = [
        python_version,
        "-m",
        "manim",
        path_basic_scene,
        "SquareToCircle",
        "-l",
        "--log_to_file",
        "--media_dir",
        str(tmp_path),
    ]
    _, err, exitcode = capture(command)
    assert exitcode == 0, err.decode()
