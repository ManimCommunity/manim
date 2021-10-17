import json
import os
import pathlib
import subprocess
from functools import wraps

import pytest
from _pytest.fixtures import FixtureRequest

from ..helpers.video_utils import (
    get_config_from_video,
    get_dir_index,
    get_section_meta,
    save_control_data_from_video,
)


def _load_video_data(path_to_data):
    with open(path_to_data) as f:
        return json.load(f)


def _check_video_data(path_control_data, path_to_video_generated):
    # movie file specification
    path_to_sections_generated = os.path.join(
        pathlib.Path(path_to_video_generated).parent.absolute(),
        "sections",
    )
    control_data = _load_video_data(path_control_data)
    config_generated = get_config_from_video(path_to_video_generated)
    config_expected = control_data["config"]
    diff_keys = [
        d1[0]
        for d1, d2 in zip(config_expected.items(), config_generated.items())
        if d1[1] != d2[1]
    ]
    # \n does not work in f-strings.
    newline = "\n"
    assert (
        len(diff_keys) == 0
    ), f"Config don't match:\n{newline.join([f'For {key}, got {config_generated[key]}, expected : {config_expected[key]}.' for key in diff_keys])}"

    # sections directory index
    section_index_generated = set(get_dir_index(path_to_sections_generated))
    section_index_expected = set(control_data["section_index"])
    unexpectedly_generated = section_index_generated - section_index_expected
    ungenerated_expected = section_index_expected - section_index_generated
    if len(unexpectedly_generated) or len(ungenerated_expected):
        dif = [
            f"'{dif}' got unexpectedly generated" for dif in unexpectedly_generated
        ] + [f"'{dif}' didn't get generated" for dif in ungenerated_expected]
        raise AssertionError(f"Sections don't match:\n{newline.join(dif)}")

    scene_name = "".join(os.path.basename(path_to_video_generated).split(".")[:-1])
    path_to_section_meta_generated = os.path.join(
        path_to_sections_generated, f"{scene_name}.json"
    )
    section_meta_generated = get_section_meta(path_to_sections_generated)
    section_meta_expected = control_data["section_meta"]

    # sections metadata file
    for section_generated, section_expected in zip(
        section_meta_generated, section_meta_expected
    ):
        if section_generated["name"] != section_expected["name"]:
            raise AssertionError(
                f"Section {section_generated} doesn't have the expected name '{section_expected['name']}'"
            )
        if section_generated["type"] != section_expected["type"]:
            raise AssertionError(
                f"Section {section_generated} doesn't have the expected type '{section_expected['type']}'"
            )
        if section_generated["video"] != section_expected["video"]:
            raise AssertionError(
                f"Section {section_generated} doesn't have the expected path to video '{section_expected['video']}'"
            )


def video_comparison(control_data_file, scene_path_from_media_dir):
    """Decorator used for any test that needs to check a rendered scene/video.

    Parameters
    ----------
    control_data_file : :class:`str`
        Name of the control data file, i.e. the .json containing all the pre-rendered references of the scene tested.
        .. warning:: You don't have to pass the path here.

    scene_path_from_media_dir : :class:`str`
        The path of the scene generated, from the media dir. Example: /videos/1080p60/SquareToCircle.mp4.

    See Also
    --------
    tests/helpers/video_utils.py : create control data
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # NOTE : Every args goes seemingly in kwargs instead of args; this is perhaps Pytest.
            result = f(*args, **kwargs)
            tmp_path = kwargs["tmp_path"]
            tests_directory = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)),
            )
            path_control_data = os.path.join(
                tests_directory,
                "control_data",
                "videos_data",
                control_data_file,
            )
            path_video_generated = tmp_path / scene_path_from_media_dir
            if not os.path.exists(path_video_generated):
                for parent in reversed(path_video_generated.parents):
                    if not parent.exists():
                        pytest.fail(
                            f"'{parent.name}' does not exist in '{parent.parent}' (which exists). ",
                        )
                        break
            # TODO: use when pytest --set_test option
            # save_control_data_from_video(
            #     path_video_generated, control_data_file[:-5]
            # )
            _check_video_data(path_control_data, str(path_video_generated))
            return result

        return wrapper

    return decorator
