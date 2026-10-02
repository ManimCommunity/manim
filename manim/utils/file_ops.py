"""Utility functions for interacting with the file system."""

from __future__ import annotations

__all__ = [
    "add_extension_if_not_present",
    "guarantee_existence",
    "guarantee_empty_existence",
    "seek_full_path_from_defaults",
    "modify_atime",
    "open_file",
    "ensure_executable",
]

import os
import platform
import shutil
import subprocess as sp
import time
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manim.typing import StrPath

    from ..scene.scene_file_writer import SceneFileWriter

from manim import __version__, config, logger

from .. import console


def ensure_executable(path_to_exe: Path) -> bool:
    if path_to_exe.parent == Path("."):
        executable: StrPath | None = shutil.which(path_to_exe.stem)
        if executable is None:
            return False
    else:
        executable = path_to_exe
    return os.access(executable, os.X_OK)


def add_extension_if_not_present(file_name: Path, extension: str) -> Path:
    if file_name.suffix != extension:
        return file_name.with_suffix(file_name.suffix + extension)
    else:
        return file_name


def add_version_before_extension(file_name: Path) -> Path:
    return file_name.with_name(
        f"{file_name.stem}_ManimCE_v{__version__}{file_name.suffix}"
    )


def guarantee_existence(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve(strict=True)


def guarantee_empty_existence(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve(strict=True)


def seek_full_path_from_defaults(
    file_name: StrPath, default_dir: Path, extensions: list[str]
) -> Path:
    possible_paths = [Path(file_name).expanduser()]
    possible_paths += [
        Path(default_dir) / f"{file_name}{extension}" for extension in ["", *extensions]
    ]
    for path in possible_paths:
        if path.exists():
            return path
    error = (
        f"From: {Path.cwd()}, could not find {file_name} at either "
        f"of these locations: {list(map(str, possible_paths))}"
    )
    raise OSError(error)


def modify_atime(file_path: str) -> None:
    """Will manually change the accessed time (called `atime`) of the file, as on a lot of OS the accessed time refresh is disabled by default.

    Parameters
    ----------
    file_path
        The path of the file.
    """
    os.utime(file_path, times=(time.time(), Path(file_path).stat().st_mtime))


def open_file(file_path: Path, in_browser: bool = False) -> None:
    current_os = platform.system()
    if current_os == "Windows":
        # os.startfile is only available on Windows, so use getattr to keep
        # static analysis platform-independent.
        startfile = getattr(os, "startfile", None)
        if startfile is None:
            raise OSError("os.startfile is unavailable on this Windows system")
        startfile(file_path if not in_browser else file_path.parent)
    else:
        if current_os == "Linux":
            commands = ["xdg-open"]
            file_path = file_path if not in_browser else file_path.parent
        elif current_os.startswith("CYGWIN"):
            commands = ["cygstart"]
            file_path = file_path if not in_browser else file_path.parent
        elif current_os == "Darwin":
            commands = ["open"] if not in_browser else ["open", "-R"]
        else:
            raise OSError("Unable to identify your operating system...")

        # check after so that file path is set correctly
        if config.preview_command:
            commands = [config.preview_command]
        commands.append(str(file_path))
        sp.run(commands)


def open_media_file(
    file_writer: SceneFileWriter,
    *,
    preview: bool,
    show_in_file_browser: bool,
) -> None:
    final_file_path = getattr(file_writer, "final_file_path", None)
    if final_file_path is None:
        logger.warning("No media artifact is available to open.")
        return
    file_paths = [final_file_path]

    for file_path in file_paths:
        if show_in_file_browser:
            open_file(file_path, True)
        if preview:
            open_file(file_path, False)

            logger.info(f"Previewed File at: '{file_path}'")


def get_template_names() -> list[str]:
    """Returns template names from the templates directory.

    Returns
    -------
        :class:`list`
    """
    template_path = Path.resolve(Path(__file__).parent.parent / "templates")
    return [template_name.stem for template_name in template_path.glob("*.mtp")]


def get_template_path() -> Path:
    """Returns the Path of templates directory.

    Returns
    -------
        :class:`Path`
    """
    return Path.resolve(Path(__file__).parent.parent / "templates")


def add_import_statement(file: Path) -> None:
    """Prepends an import statement in a file

    Parameters
    ----------
        file
    """
    with file.open("r+") as f:
        import_line = "from manim import *"
        content = f.read()
        f.seek(0)
        f.write(import_line + "\n" + content)


def copy_template_files(
    project_dir: Path = Path("."), template_name: str = "Default"
) -> None:
    """Copies template files from templates dir to project_dir.

    Parameters
    ----------
        project_dir
            Path to project directory.
        template_name
            Name of template.
    """
    template_cfg_path = Path.resolve(
        Path(__file__).parent.parent / "templates/template.cfg",
    )
    template_scene_path = Path.resolve(
        Path(__file__).parent.parent / f"templates/{template_name}.mtp",
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
