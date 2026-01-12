from __future__ import annotations

import itertools
import json
import os
from functools import wraps
from pathlib import Path

import pytest


def _check_logs(reference_logfile_path: Path, generated_logfile_path: Path) -> None:
    with reference_logfile_path.open() as reference_logfile:
        reference_logs = reference_logfile.readlines()
    with generated_logfile_path.open() as generated_logfile:
        generated_logs = generated_logfile.readlines()
    diff = abs(len(reference_logs) - len(generated_logs))
    if len(reference_logs) != len(generated_logs):
        msg_assert = ""
        if len(reference_logs) > len(generated_logs):
            msg_assert += f"Logs generated are SHORTER than the expected logs. There are {diff} extra logs.\n"
            msg_assert += "Last log of the generated log is : \n"
            msg_assert += generated_logs[-1]
        else:
            msg_assert += f"Logs generated are LONGER than the expected logs.\n There are {diff} extra logs :\n"
            for log in generated_logs[len(reference_logs) :]:
                msg_assert += log
        msg_assert += f"\nPath of reference log: {reference_logfile}\nPath of generated logs: {generated_logfile}"
        pytest.fail(msg_assert)

    for index, ref, gen in zip(itertools.count(), reference_logs, generated_logs):
        # As they are string, we only need to check if they are equal. If they are not, we then compute a more precise difference, to debug.
        if ref == gen:
            continue
        ref_log = json.loads(ref)
        gen_log = json.loads(gen)
        diff_keys = [
            d1[0]
            for d1, d2 in zip(ref_log.items(), gen_log.items(), strict=False)
            if d1[1] != d2[1]
        ]
        # \n and \t don't not work in f-strings.
        newline = "\n"
        tab = "\t"
        assert len(diff_keys) == 0, (
            f"Logs don't match at {index} log. : \n{newline.join([f'In {key} field, got -> {newline}{tab}{repr(gen_log[key])}. {newline}Expected : -> {newline}{tab}{repr(ref_log[key])}.' for key in diff_keys])}"
            + f"\nPath of reference log: {reference_logfile}\nPath of generated logs: {generated_logfile}"
        )


def logs_comparison(
    control_data_file: str | os.PathLike, log_path_from_media_dir: str | os.PathLike
):
    """Decorator used for any test that needs to check logs.

    Parameters
    ----------
    control_data_file
        Name of the control data file, i.e. .log that will be compared to the outputted logs.
        .. warning:: You don't have to pass the path here.
        .. example:: "SquareToCircleWithLFlag.log"

    log_path_from_media_dir
        The path of the .log generated, from the media dir. Example: /logs/Square.log.

    Returns
    -------
    Callable[[Any], Any]
        The test wrapped with which we are going to make the comparison.
    """
    control_data_file = Path(control_data_file)
    log_path_from_media_dir = Path(log_path_from_media_dir)

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # NOTE : Every args goes seemingly in kwargs instead of args; this is perhaps Pytest.
            result = f(*args, **kwargs)
            tmp_path = kwargs["tmp_path"]
            tests_directory = Path(__file__).absolute().parent.parent
            control_data_path = (
                tests_directory / "control_data" / "logs_data" / control_data_file
            )
            path_log_generated = tmp_path / log_path_from_media_dir
            # The following will say precisely which subdir does not exist.
            if not path_log_generated.exists():
                for parent in reversed(path_log_generated.parents):
                    if not parent.exists():
                        pytest.fail(
                            f"'{parent.name}' does not exist in '{parent.parent}' (which exists). ",
                        )
                        break
            _check_logs(control_data_path, path_log_generated)
            return result

        return wrapper

    return decorator
