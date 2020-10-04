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
import numpy as np
import time
import re
import subprocess as sp


def add_extension_if_not_present(file_name, extension):
    # This could conceivably be smarter about handling existing differing extensions
    if file_name[-len(extension) :] != extension:
        return file_name + extension
    else:
        return file_name


def guarantee_existence(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.abspath(path)


def seek_full_path_from_defaults(file_name, default_dir, extensions):
    possible_paths = [file_name]
    possible_paths += [
        os.path.join(default_dir, file_name + extension)
        for extension in ["", *extensions]
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise IOError("File {} not Found".format(file_name))


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
