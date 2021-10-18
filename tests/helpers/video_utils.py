"""Helpers for dev to set up new tests that use videos."""

import json
import os
import pathlib
from typing import Any, Dict, List

from manim import get_dir_layout, get_video_metadata, logger


def get_section_dir_layout(dirpath: str) -> List[str]:
    """Return a list of all files in the sections directory."""
    # test if sections have been created in the first place, doesn't work with multiple scene but this isn't an issue with tests
    if not os.path.isdir(dirpath):
        return []
    files = get_dir_layout(dirpath)
    # indicate that the sections directory has been created
    files.append(".")
    return files


def get_section_index(metapath: str) -> List[Dict[str, Any]]:
    """Return content of sections index file."""
    parent_folder = pathlib.Path(metapath).parent.absolute()
    # test if sections have been created in the first place
    if not os.path.isdir(parent_folder):
        return []
    with open(metapath) as file:
        index = json.load(file)
    return index


def save_control_data_from_video(path_to_video: str, name: str) -> None:
    """Helper used to set up a new test that will compare videos.

    This will create a new ``.json`` file in ``control_data/videos_data`` that contains:
    - the name of the video,
    - the metadata of the video, like fps and resolution and
    - the paths of all files in the sections subdirectory (like section videos).

    Refer to the documentation for more information.

    Parameters
    ----------
    path_to_video : :class:`str`
        Path to the video to extract information from.
    name : :class:`str`
        Name of the test. The .json file will be named with it.

    See Also
    --------
    tests/utils/video_tester.py : read control data and compare with output of test
    """
    path_to_sections = os.path.join(
        pathlib.Path(path_to_video).parent.absolute(), "sections"
    )
    tests_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_control_data = os.path.join(tests_directory, "control_data", "videos_data")
    # this is the name of the section used in the test, not the name of the test itself, it can be found as a parameter of this function
    scene_name = "".join(os.path.basename(path_to_video).split(".")[:-1])

    movie_metadata = get_video_metadata(path_to_video)
    section_dir_layout = get_section_dir_layout(path_to_sections)
    section_index = get_section_index(
        os.path.join(path_to_sections, f"{scene_name}.json")
    )
    data = {
        "name": name,
        "movie_metadata": movie_metadata,
        "section_dir_layout": section_dir_layout,
        "section_index": section_index,
    }
    path_saved = os.path.join(path_control_data, f"{name}.json")
    with open(path_saved, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Data for {name} saved in {path_saved}")
