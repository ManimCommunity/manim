import json
import os
import pathlib
from functools import wraps

from manim import get_video_metadata

from ..assert_utils import assert_shallow_dict_compare
from ..helpers.video_utils import (
    get_dir_index,
    get_section_meta,
    save_control_data_from_video,
)


def load_video_data(path_to_data):
    with open(path_to_data) as f:
        return json.load(f)


def check_video_data(path_control_data, path_to_video_generated):
    # movie file specification
    path_to_sections_generated = os.path.join(
        pathlib.Path(path_to_video_generated).parent.absolute(),
        "sections",
    )
    control_data = load_video_data(path_control_data)
    config_generated = get_video_metadata(path_to_video_generated)
    config_expected = control_data["config"]

    assert_shallow_dict_compare(
        config_generated, config_expected, "Movie file specification mismatch:"
    )

    # sections directory index
    section_index_generated = set(get_dir_index(path_to_sections_generated))
    section_index_expected = set(control_data["section_index"])
    unexpectedly_generated = section_index_generated - section_index_expected
    ungenerated_expected = section_index_expected - section_index_generated
    if len(unexpectedly_generated) or len(ungenerated_expected):
        dif = [
            f"'{dif}' got unexpectedly generated" for dif in unexpectedly_generated
        ] + [f"'{dif}' didn't get generated" for dif in ungenerated_expected]
        mismatch = "\n".join(dif)
        raise AssertionError(f"Sections don't match:\n{mismatch}")

    scene_name = "".join(os.path.basename(path_to_video_generated).split(".")[:-1])
    path_to_section_meta_generated = os.path.join(
        path_to_sections_generated, f"{scene_name}.json"
    )
    section_meta_generated = get_section_meta(path_to_section_meta_generated)
    section_meta_expected = control_data["section_meta"]

    # sections metadata file
    if len(section_meta_generated) != len(section_meta_expected):
        raise AssertionError(
            f"expected {len(section_meta_expected)} sections ({', '.join([el['name'] for el in section_meta_expected])}), but {len(section_meta_generated)} ({', '.join([el['name'] for el in section_meta_generated])}) got generated (in '{path_to_section_meta_generated}')"
        )
    # check individual sections
    for section_generated, section_expected in zip(
        section_meta_generated, section_meta_expected
    ):
        assert_shallow_dict_compare(
            section_generated,
            section_expected,
            # using json to pretty print dicts
            f"Section {json.dumps(section_generated, indent=4)} (in '{path_to_section_meta_generated}') doesn't match expected Section (in '{json.dumps(section_expected, indent=4)}'):",
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
                        raise AssertionError(
                            f"'{parent.name}' does not exist in '{parent.parent}' (which exists). ",
                        )
                        break
            # TODO: use when pytest --set_test option
            # save_control_data_from_video(path_video_generated, control_data_file[:-5])
            check_video_data(path_control_data, str(path_video_generated))
            return result

        return wrapper

    return decorator
