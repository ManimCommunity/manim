"""Helpers for dev to set up new tests that use videos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from manim import get_dir_layout, get_video_metadata, logger


def get_section_dir_layout(dirpath: Path) -> list[str]:
    """Return a list of all files in the sections directory."""
    # test if sections have been created in the first place, doesn't work with multiple scene but this isn't an issue with tests
    if not dirpath.is_dir():
        return []
    files = list(get_dir_layout(dirpath))
    # indicate that the sections directory has been created
    files.append(".")
    return files


def get_section_index(metapath: Path) -> list[dict[str, Any]]:
    """Return content of sections index file."""
    parent_folder = metapath.parent.absolute()
    # test if sections have been created in the first place
    if not parent_folder.is_dir():
        return []
    with metapath.open() as file:
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
    orig_path_to_sections = Path(path_to_video)
    path_to_sections = orig_path_to_sections.parent.absolute() / "sections"
    tests_directory = Path(__file__).absolute().parent.parent
    path_control_data = Path(tests_directory) / "control_data" / "videos_data"
    # this is the name of the section used in the test, not the name of the test itself, it can be found as a parameter of this function
    scene_name = orig_path_to_sections.stem

    movie_metadata = get_video_metadata(path_to_video)
    section_dir_layout = get_section_dir_layout(path_to_sections)
    section_index = get_section_index(path_to_sections / f"{scene_name}.json")

    data = {
        "name": name,
        "movie_metadata": movie_metadata,
        "section_dir_layout": section_dir_layout,
        "section_index": section_index,
    }
    path_saved = Path(path_control_data) / f"{name}.json"
    with open(path_saved, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Data for {name} saved in {path_saved}")
