import json
import os
import pathlib
from functools import wraps

from manim import get_video_metadata

from ..assert_utils import assert_shallow_dict_compare
from ..helpers.video_utils import (
    get_section_dir_layout,
    get_section_index,
    save_control_data_from_video,
)


def load_control_data(path_to_data):
    with open(path_to_data) as f:
        return json.load(f)


def check_video_data(path_control_data, path_video_gen):
    """Compare control data with generated output.
    Used abbreviations:
        exp  -> expected
        gen  -> generated
        sec  -> section
        meta -> metadata
    """
    # movie file specification
    path_sec_gen = os.path.join(
        pathlib.Path(path_video_gen).parent.absolute(),
        "sections",
    )
    control_data = load_control_data(path_control_data)
    movie_meta_gen = get_video_metadata(path_video_gen)
    movie_meta_exp = control_data["movie_metadata"]

    assert_shallow_dict_compare(
        movie_meta_gen, movie_meta_exp, "Movie file metadata mismatch:"
    )

    # sections directory layout
    sec_dir_layout_gen = set(get_section_dir_layout(path_sec_gen))
    sec_dir_layout_exp = set(control_data["section_dir_layout"])

    unexp_gen = sec_dir_layout_gen - sec_dir_layout_exp
    ungen_exp = sec_dir_layout_exp - sec_dir_layout_gen
    if len(unexp_gen) or len(ungen_exp):
        dif = [f"'{dif}' got unexpectedly generated" for dif in unexp_gen] + [
            f"'{dif}' didn't get generated" for dif in ungen_exp
        ]
        mismatch = "\n".join(dif)
        raise AssertionError(f"Sections don't match:\n{mismatch}")

    # sections index file
    scene_name = "".join(os.path.basename(path_video_gen).split(".")[:-1])
    path_sec_index_gen = os.path.join(path_sec_gen, f"{scene_name}.json")
    sec_index_gen = get_section_index(path_sec_index_gen)
    sec_index_exp = control_data["section_index"]

    if len(sec_index_gen) != len(sec_index_exp):
        raise AssertionError(
            f"expected {len(sec_index_exp)} sections ({', '.join([el['name'] for el in sec_index_exp])}), but {len(sec_index_gen)} ({', '.join([el['name'] for el in sec_index_gen])}) got generated (in '{path_sec_index_gen}')"
        )
    # check individual sections
    for sec_gen, sec_exp in zip(sec_index_gen, sec_index_exp):
        assert_shallow_dict_compare(
            sec_gen,
            sec_exp,
            # using json to pretty print dicts
            f"Section {json.dumps(sec_gen, indent=4)} (in '{path_sec_index_gen}') doesn't match expected Section (in '{json.dumps(sec_exp, indent=4)}'):",
        )


def video_comparison(control_data_file, scene_path_from_media_dir):
    """Decorator used for any test that needs to check a rendered scene/video.

    .. warning::
        The directories, such as the movie dir or sections dir, are expected to abide by the default.
        This requirement could be dropped if the manim config were to be accessible from ``wrapper`` like in ``frames_comparison.py``.

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
            path_video_gen = tmp_path / scene_path_from_media_dir
            if not os.path.exists(path_video_gen):
                for parent in reversed(path_video_gen.parents):
                    if not parent.exists():
                        raise AssertionError(
                            f"'{parent.name}' does not exist in '{parent.parent}' (which exists). ",
                        )
            # TODO: use when pytest --set_test option
            # save_control_data_from_video(path_video_gen, control_data_file[:-5])
            check_video_data(path_control_data, str(path_video_gen))
            return result

        return wrapper

    return decorator
