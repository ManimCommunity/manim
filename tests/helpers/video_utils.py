"""Helpers for dev to set up new tests that use videos."""

import json
import os
import pathlib
from typing import Dict, List

from manim import config, logger

from ..utils.commands import capture


def get_config_from_video(path_to_video: str):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,duration,avg_frame_rate,codec_name",
        "-print_format",
        "json",
        path_to_video,
    ]
    config, err, exitcode = capture(command)
    assert exitcode == 0, f"FFprobe error: {err}"
    return json.loads(config)["streams"][0]


def get_dir_index(dirpath: str) -> List[str]:
    # test if sections have been created in the first place, doesn't work with multiple scene but this isn't an issue with tests
    if not os.path.isdir(dirpath):
        return []
    # indicate that the sections directory has been created
    index_files: List[str] = ["."]
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            index_files.append(f"{os.path.relpath(os.path.join(root, file), dirpath)}")
    return index_files


def get_section_meta(metapath: str) -> List[Dict[str, str]]:
    # test if sections have been created in the first place
    if not os.path.isdir(pathlib.Path(metapath).parent.absolute()):
        return []
    with open(metapath) as file:
        sections = json.load(file)
    return sections


def save_control_data_from_video(path_to_video: str, name: str) -> None:
    """Helper used to set up a new test that will compare videos.
    This will create a new .json file in control_data/videos_data that contains:
    - the name of the video,
    - the specification (called 'config' in the code) of the video, like fps and resolution and
    - the paths of all files in the sections subdirectory (like section videos).
    Refer to the wiki for more information.

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
    # TODO: "sections" might be changed in the future
    path_to_sections = os.path.join(
        pathlib.Path(path_to_video).parent.absolute(), "sections"
    )
    tests_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_control_data = os.path.join(tests_directory, "control_data", "videos_data")
    # TODO: "config" might be a confusing name for the specification of the movie, it isn't referring to the Manim config after all
    config = get_config_from_video(path_to_video)
    section_index = get_dir_index(path_to_sections)
    # this is the name of the section used in the test, not the name of the test itself, it can be found as a parameter
    scene_name = "".join(os.path.basename(path_to_video).split(".")[:-1])
    section_meta = get_section_meta(
        os.path.join(path_to_sections, f"{scene_name}.json")
    )
    data = {
        "name": name,
        "config": config,
        "section_index": section_index,
        "section_meta": section_meta,
    }
    path_saved = os.path.join(path_control_data, f"{name}.json")
    with open(path_saved, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Data for {name} saved in {path_saved}")
