"""Helpers functions for devs to set up new graphical-units data."""


from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from manim import config, logger


def set_test_scene(scene_object, module_name):
    """Function used to set up the test data for a new feature. This will basically set up a pre-rendered frame for a scene. This is meant to be used only
    when setting up tests. Please refer to the wiki.

    Parameters
    ----------
    scene_object : :class:`~.Scene`
        The scene with which we want to set up a new test.
    module_name : :class:`str`
        The name of the module in which the functionality tested is contained. For example, ``Write`` is contained in the module ``creation``. This will be used in the folder architecture
        of ``/tests_data``.

    Examples
    --------
    Normal usage::
        set_test_scene(DotTest, "geometry")

    """
    config["write_to_movie"] = False
    config["disable_caching"] = True
    config["format"] = "png"
    config["pixel_height"] = 480
    config["pixel_width"] = 854
    config["frame_rate"] = 15

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        config["text_dir"] = temp_path / "text"
        config["tex_dir"] = temp_path / "tex"
        scene = scene_object(skip_animations=True)
        scene.render()
        data = scene.renderer.get_frame()

    assert not np.all(
        data == np.array([0, 0, 0, 255]),
    ), f"Control data generated for {str(scene)} only contains empty pixels."
    assert data.shape == (480, 854, 4)
    tests_directory = Path(__file__).absolute().parent.parent
    path_control_data = Path(tests_directory) / "control_data" / "graphical_units_data"
    path = Path(path_control_data) / module_name
    if not path.is_dir():
        path.mkdir(parents=True)
    np.savez_compressed(path / str(scene), frame_data=data)
    logger.info(f"Test data for {str(scene)} saved in {path}\n")
