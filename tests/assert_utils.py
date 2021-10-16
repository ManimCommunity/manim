import typing
from os import PathLike
from pathlib import Path
from pprint import pformat


def assert_file_exists(filepath: typing.Union[str, PathLike]) -> None:
    """Assert if filepath points to an existing file. Print all the elements (files and dir) of the parent dir of the given filepath.

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
        message = f"{path.absolute()} is not a file. Other elements in the parent directory are \n{pformat([path.name for path in list(path.parent.iterdir())])}"
        raise AssertionError(message)


def assert_dir_exists(dirpath: typing.Union[str, PathLike]) -> None:
    """Assert if directory exists

    Parameters
    ----------
    filepath
        Filepath to check.

    Raises
    ------
    AssertionError
        If dirpath doesn't point to a directory (if the file does exist or it's a file).
    """
    path = Path(dirpath)
    if not path.is_dir():
        message = f"{path.absolute()} is not a directory. Other elements in the parent directory are \n{pformat([path.name for path in list(path.parent.iterdir())])}"
        raise AssertionError(message)


def assert_file_not_exists(filepath: typing.Union[str, PathLike]) -> None:
    """Assert if filepath doesn't point to an existing file. Print all the elements (files and dir) of the parent dir of the given filepath.

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
        message = f"{path.absolute()} is a file. Other elements in the parent directory are \n{pformat([path.name for path in list(path.parent.iterdir())])}"
        raise AssertionError(message)
