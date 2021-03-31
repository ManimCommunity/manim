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
