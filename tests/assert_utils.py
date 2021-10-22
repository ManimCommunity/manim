import os
from pathlib import Path
from pprint import pformat
from typing import Dict, List, Union


def assert_file_exists(filepath: Union[str, os.PathLike]) -> None:
    """Assert that filepath points to an existing file. Print all the elements (files and dir) of the parent dir of the given filepath.

    This is mostly to have better assert message than using a raw assert os.path.isfile(filepath).

    Parameters
    ----------
    filepath
        Filepath to check.

    Raises
    ------
    AssertionError
        If filepath does not point to a file (if the file does not exist or it's a dir).
    """
    path = Path(filepath)
    if not path.is_file():
        elems = pformat([path.name for path in list(path.parent.iterdir())])
        message = f"{path.absolute()} is not a file. Other elements in the parent directory are \n{elems}"
        raise AssertionError(message)


def assert_dir_exists(dirpath: Union[str, os.PathLike]) -> None:
    """Assert that directory exists.

    Parameters
    ----------
    dirpath
        Path to directory to check.

    Raises
    ------
    AssertionError
        If dirpath does not point to a directory (if the file does exist or it's a file).
    """
    path = Path(dirpath)
    if not path.is_dir():
        elems = pformat([path.name for path in list(path.parent.iterdir())])
        message = f"{path.absolute()} is not a directory. Other elements in the parent directory are \n{elems}"
        raise AssertionError(message)


def assert_dir_filled(dirpath: Union[str, os.PathLike]) -> None:
    """Assert that directory exists and contains at least one file or directory (or file like objects like symlinks on Linux).

    Parameters
    ----------
    dirpath
        Path to directory to check.

    Raises
    ------
    AssertionError
        If dirpath does not point to a directory (if the file does exist or it's a file) or the directory is empty.
    """
    if len(os.listdir(dirpath)) == 0:
        raise AssertionError(f"{dirpath} is an empty directory.")


def assert_file_not_exists(filepath: Union[str, os.PathLike]) -> None:
    """Assert that filepath does not point to an existing file. Print all the elements (files and dir) of the parent dir of the given filepath.

    This is mostly to have better assert message than using a raw assert os.path.isfile(filepath).

    Parameters
    ----------
    filepath
        Filepath to check.

    Raises
    ------
    AssertionError
        If filepath does point to a file.
    """
    path = Path(filepath)
    if path.is_file():
        elems = pformat([path.name for path in list(path.parent.iterdir())])
        message = f"{path.absolute()} is a file. Other elements in the parent directory are \n{elems}"
        raise AssertionError(message)


def assert_dir_not_exists(dirpath: Union[str, os.PathLike]) -> None:
    """Assert that directory does not exist.

    Parameters
    ----------
    dirpath
        Path to directory to check.

    Raises
    ------
    AssertionError
        If dirpath points to a directory.
    """
    path = Path(dirpath)
    if path.is_dir():
        elems = pformat([path.name for path in list(path.parent.iterdir())])
        message = f"{path.absolute()} is a directory. Other elements in the parent directory are \n{elems}"
        raise AssertionError(message)


def assert_shallow_dict_compare(a: Dict, b: Dict, message_start: str) -> None:
    """Assert that Directories ``a`` and ``b`` are the same.

    ``b`` is treated as the expected values that ``a`` shall abide by.
    Print helpful error with custom message start.
    """
    mismatch: List[str] = []

    for b_key, b_value in b.items():
        if b_key not in a:
            mismatch.append(f"Missing item {b_key}: {b_value}")
        elif b_value != a[b_key]:
            mismatch.append(f"For {b_key} got {a[b_key]}, expected {b_value}")

    for a_key, a_value in a.items():
        if a_key not in b:
            mismatch.append(f"Extraneous item {a_key}: {a_value}")

    mismatch_str = "\n".join(mismatch)
    assert len(mismatch) == 0, f"{message_start}\n{mismatch_str}"
