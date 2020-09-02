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
        tmp_path,
    ]
    out, err, exitcode = capture(command)
    print(out)
    assert exitcode == 0, err.decode()


# def test_logging_to_file(tmp_path, python_version):
#     """Test logging Terminal output to a log file.
#     As some data will differ with each log (the timestamps, file paths, line nums etc)
#     a regex substitution has been employed to replace the strings that may change with
#     whitespace.
#     """
#     path_basic_scene = os.path.join("tests", "test_logging", "basic_scenes.py")
#     path_output = os.path.join(tmp_path, "media_temp")
#     os.makedirs(tmp_path, exist_ok=True)
#     command = [
#         python_version,
#         "-m",
#         "manim",
#         path_basic_scene,
#         "SquareToCircle",
#         "-l",
#         "--log_to_file",
#         "--log_dir",
#         os.path.join(path_output, "logs"),
#         "--media_dir",
#         path_output,
#         "-v",
#         "DEBUG",
#     ]
#     out, err, exitcode = capture(command, use_shell=True)
#     log_file_path = os.path.join(path_output, "logs", "SquareToCircle.log")
#     assert exitcode == 0, err.decode()
#     assert os.path.exists(log_file_path), err.decode()
#     if sys.platform.startswith("win32") or sys.platform.startswith("cygwin"):
#         enc = "Windows-1252"
#     else:
#         enc = "utf-8"
#     with open(log_file_path, encoding=enc) as logfile:
#         logs = logfile.read()
#     # The following regex pattern selects file paths and all numbers.
#     pattern = r"(\['[A-Z]?:?[\/\\].*cfg'])|([A-Z]?:?[\/\\].*mp4)|(\d+)"

#     logs = re.sub(pattern, lambda m: " " * len((m.group(0))), logs)
#     with open(
#         os.path.join(os.path.dirname(__file__), "expected.txt"), "r"
#     ) as expectedfile:
#         expected = re.sub(
#             pattern, lambda m: " " * len((m.group(0))), expectedfile.read()
#         )
#     assert logs == expected, logs
