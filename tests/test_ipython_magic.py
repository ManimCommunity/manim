from __future__ import annotations

import re

from manim.utils.ipython_magic import _generate_file_name


def test_jupyter_file_naming(config):
    """Check the format of file names for jupyter"""
    scene_name = "SimpleScene"
    expected_pattern = r"[0-9a-zA-Z_]+[@_-]\d\d\d\d-\d\d-\d\d[@_-]\d\d-\d\d-\d\d"
    config.scene_names = [scene_name]
    file_name = _generate_file_name()
    match = re.match(expected_pattern, file_name)
    assert scene_name in file_name, (
        "Expected file to contain " + scene_name + " but got " + file_name
    )
    assert match, "file name does not match expected pattern " + expected_pattern


def test_jupyter_file_output(tmp_path, config):
    """Check the jupyter file naming is valid and can be created"""
    scene_name = "SimpleScene"
    config.scene_names = [scene_name]
    file_name = _generate_file_name()
    actual_path = tmp_path.with_name(file_name)
    with actual_path.open("w") as outfile:
        outfile.write("")
        assert actual_path.exists()
        assert actual_path.is_file()
