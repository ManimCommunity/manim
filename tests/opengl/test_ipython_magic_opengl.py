import re

from manim import config, tempconfig
from manim.utils.ipython_magic import _generate_file_name


def test_jupyter_file_naming():
    """Check the format of file names for jupyter"""
    scene_name = "SimpleScene"
    expected_pattern = r"[0-9a-zA-Z_]+[@_-]\d\d\d\d-\d\d-\d\d[@_-]\d\d-\d\d-\d\d"
    current_renderer = config.renderer
    with tempconfig({"scene_names": [scene_name], "renderer": "opengl"}):
        file_name = _generate_file_name()
        match = re.match(expected_pattern, file_name)
        assert scene_name in file_name, (
            "Expected file to contain " + scene_name + " but got " + file_name
        )
        assert match, "file name does not match expected pattern " + expected_pattern
    # needs manually set back to avoid issues across tests
    config.renderer = current_renderer


def test_jupyter_file_output(tmp_path):
    """Check the jupyter file naming is valid and can be created"""
    scene_name = "SimpleScene"
    current_renderer = config.renderer
    with tempconfig({"scene_names": [scene_name], "renderer": "opengl"}):
        file_name = _generate_file_name()
        actual_path = tmp_path.with_name(file_name)
        with open(actual_path, "w") as outfile:
            outfile.write("")
            assert actual_path.exists()
            assert actual_path.is_file()
    # needs manually set back to avoid issues across tests
    config.renderer = current_renderer
