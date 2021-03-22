"""Utility functions for interacting with the file system."""

__all__ = [
    "add_extension_if_not_present",
    "guarantee_existence",
    "seek_full_path_from_defaults",
    "modify_atime",
    "open_file",
]


import os
import platform
import time
import subprocess as sp
from manim import config, logger
from pathlib import Path

from manim import __version__


def add_extension_if_not_present(file_name, extension):
    if file_name.suffix != extension:
        return file_name.with_suffix(extension)
    else:
        return file_name


def add_version_before_extension(file_name):
    file_name = Path(file_name)
    path, name, suffix = file_name.parent, file_name.stem, file_name.suffix
    return Path(path, f"{name}_ManimCE_v{__version__}{suffix}")


def guarantee_existence(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.abspath(path)


def seek_full_path_from_defaults(file_name, default_dir, extensions):
    possible_paths = [file_name]
    possible_paths += [
        Path(default_dir) / f"{file_name}{extension}" for extension in ["", *extensions]
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    error = f"From: {os.getcwd()}, could not find {file_name} at either of these locations: {possible_paths}"
    raise IOError(error)


def modify_atime(file_path):
    """Will manually change the accessed time (called `atime`) of the file, as on a lot of OS the accessed time refresh is disabled by default.

    Parameters
    ----------
    file_path : :class:`str`
        The path of the file.
    """
    os.utime(file_path, times=(time.time(), os.path.getmtime(file_path)))


def open_file(file_path, in_browser=False):
    current_os = platform.system()
    if current_os == "Windows":
        os.startfile(file_path if not in_browser else os.path.dirname(file_path))
    else:
        if current_os == "Linux":
            commands = ["xdg-open"]
            file_path = file_path if not in_browser else os.path.dirname(file_path)
        elif current_os.startswith("CYGWIN"):
            commands = ["cygstart"]
            file_path = file_path if not in_browser else os.path.dirname(file_path)
        elif current_os == "Darwin":
            commands = ["open"] if not in_browser else ["open", "-R"]
        else:
            raise OSError("Unable to identify your operating system...")
        commands.append(file_path)
        sp.Popen(commands)


def open_media_file(file_writer):
    file_paths = []

    if config["save_last_frame"]:
        file_paths.append(file_writer.image_file_path)
    if config["write_to_movie"] and not config["save_as_gif"]:
        file_paths.append(file_writer.movie_file_path)
    if config["save_as_gif"]:
        file_paths.append(file_writer.gif_file_path)

    for file_path in file_paths:
        if config["show_in_file_browser"]:
            open_file(file_path, True)
        if config["preview"]:
            open_file(file_path, False)

            logger.info(f"Previewed File at: {file_path}")
