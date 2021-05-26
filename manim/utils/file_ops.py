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
import subprocess as sp
import time
from pathlib import Path
from shutil import copyfile

from manim import __version__, config, logger

from .. import console


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
    if config["write_to_movie"] and not config["format"] == "gif":
        file_paths.append(file_writer.movie_file_path)
    if config["format"] == "gif":
        file_paths.append(file_writer.gif_file_path)

    for file_path in file_paths:
        if config["show_in_file_browser"]:
            open_file(file_path, True)
        if config["preview"]:
            open_file(file_path, False)

            logger.info(f"Previewed File at: {file_path}")


def get_template_names():
    """Returns template names from the templates directory.

    Returns
    -------
        :class:`list`
    """
    template_path = Path.resolve(Path(__file__).parent.parent / "templates")
    return [template_name.stem for template_name in template_path.glob("*.mtp")]


def get_template_path():
    """Returns the Path of templates directory.

    Returns
    -------
        :class:`Path`
    """
    return Path.resolve(Path(__file__).parent.parent / "templates")


def add_import_statement(file):
    """Prepends an import statement in a file

    Parameters
    ----------
        file : :class:`Path`
    """
    with open(file, "r+") as f:
        import_line = "from manim import *"
        content = f.read()
        f.seek(0, 0)
        f.write(import_line.rstrip("\r\n") + "\n" + content)


def copy_template_files(project_dir=Path("."), template_name="Default"):
    """Copies template files from templates dir to project_dir.

    Parameters
    ----------
        project_dir : :class:`Path`
            Path to project directory.
        template_name : :class:`str`
            Name of template.
    """
    template_cfg_path = Path.resolve(
        Path(__file__).parent.parent / "templates/template.cfg"
    )
    template_scene_path = Path.resolve(
        Path(__file__).parent.parent / f"templates/{template_name}.mtp"
    )

    if not template_cfg_path.exists():
        raise FileNotFoundError(f"{template_cfg_path} : file does not exist")
    if not template_scene_path.exists():
        raise FileNotFoundError(f"{template_scene_path} : file does not exist")

    copyfile(template_cfg_path, Path.resolve(project_dir / "manim.cfg"))
    console.print("\n\t[green]copied[/green] [blue]manim.cfg[/blue]\n")
    copyfile(template_scene_path, Path.resolve(project_dir / "main.py"))
    console.print("\n\t[green]copied[/green] [blue]main.py[/blue]\n")
    add_import_statement(Path.resolve(project_dir / "main.py"))
