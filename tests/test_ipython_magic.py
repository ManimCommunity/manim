import re

from manim import tempconfig
from manim.utils.ipython_magic import _generate_file_name


def test_jupyter_file_naming():
    """Check the format of file names for jupyter"""
    scene_name = "SimpleScene"
    expected_pattern = "[0-9a-zA-Z@_-]+[@_-]\d\d\d\d-\d\d-\d\d[@_-]\d\d-\d\d-\d\d"
    with tempconfig({"scene_names": ["SimpleScene"]}):
        file_name = _generate_file_name()
        match = re.match(expected_pattern, file_name)
        assert scene_name in file_name, (
            "Expected file to contain " + scene_name + " but got " + file_name
        )
        assert match, "file name does not match expected pattern " + expected_pattern
