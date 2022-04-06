"""Utility functions for interacting with the file system."""

from __future__ import annotations

__all__ = [
    "add_extension_if_not_present",
    "guarantee_existence",
    "guarantee_empty_existence",
    "seek_full_path_from_defaults",
    "modify_atime",
    "open_file",
    "is_mp4_format",
    "is_gif_format",
    "is_png_format",
    "is_webm_format",
    "is_mov_format",
    "write_to_movie",
]

import configparser
import functools
import os
import platform
import shutil
import subprocess as sp
import time
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..scene.scene_file_writer import SceneFileWriter

from manim import __version__, config, constants, logger

from .. import console
from .._config.utils import config_file_paths


@functools.lru_cache(maxsize=None)
def check_ffmpeg_exe_working(ffmpeg_exe: str | None) -> bool:
    if not ffmpeg_exe:
        return False
    if not Path(ffmpeg_exe).is_file():
        logger.info(f"{ffmpeg_exe} doesn't exists")
        return False
    logger.info(f"Checking if running '{ffmpeg_exe} -version' works...")
    op = sp.run([ffmpeg_exe, "-version"], stdout=sp.PIPE)
    if op.returncode != 0:
        logger.info(
            f"Running '{ffmpeg_exe} -version' returned non-zero exit-code {op.returncode}"
        )
        logger.info(f"stdout: {op.stdout}")
        return False
    return True


@functools.lru_cache(maxsize=None)
def check_latex_exe_working(latex_exe: str | None) -> bool:
    if not latex_exe:
        return False
    if not Path(latex_exe).exists():
        logger.info(f"{latex_exe} doesn't exists")
        return False
    logger.info(f"Checking if running '{latex_exe} --version' works...")
    op = sp.run([latex_exe, "--version"], stdout=sp.PIPE)
    if op.returncode != 0:
        logger.info(
            f"Running '{latex_exe}' returned non-zero exit-code {op.returncode}"
        )
        logger.info(f"stdout: {op.stdout}")
        return False
    return True


def prompt_user_for_choice(input_ffmpeg: bool = False, input_latex: bool = False):
    ffmpeg_from_path = shutil.which("ffmpeg")
    latex_from_path = shutil.which("latex")
    ffmpeg_exe = None
    latex_exe = None

    while input_ffmpeg:
        ffmpeg_exe = console.input(
            f"[log.message] {constants.INPUT_FFMPEG_EXECUTABLE_MESSAGE.format(DEFAULT_FFMPEG=ffmpeg_from_path)} [/log.message]"
        )
        if not ffmpeg_exe:
            ffmpeg_exe = ffmpeg_from_path
        input_ffmpeg = not check_ffmpeg_exe_working(ffmpeg_exe)
        if input_ffmpeg is True:
            logger.info(f"'{ffmpeg_exe}' doesn't seem to work")
            logger.info("Please try again.")

    while input_latex:
        latex_exe = console.input(
            f"[log.message] {constants.INPUT_LATEX_EXECUTABLE_MESSAGE.format(DEFAULT_LATEX=latex_from_path)} [/log.message]"
        )
        if not latex_exe:
            latex_exe = latex_from_path
        input_latex = not check_latex_exe_working(latex_exe)
        if input_latex is True:
            logger.info(f"'{latex_exe}' doesn't seem to work")
            logger.info("Please try again.")

    return ffmpeg_exe, latex_exe


def save_config_latex_ffmpeg(ffmpeg_exe: str = None, latex_exe: str = None) -> None:
    if ffmpeg_exe:
        _inp = console.input(
            "[log.message] Do you want to save ffmpeg executable?: (Y/N) [/log.message]"
        )
    if latex_exe:
        _inp = console.input(
            "[log.message] Do you want to save latex executable?: (Y/N) [/log.message]"
        )
    if _inp.lower() == "y":
        # Save the file in default.cfg
        library_wide, _, _ = config_file_paths()
        _parser = configparser.ConfigParser()
        with open(library_wide, encoding="utf-8") as f:
            _parser.read_file(f)

        # always create a backup for original file
        shutil.copyfile(library_wide, os.fspath(library_wide) + ".old")

        if ffmpeg_exe:
            _parser["CLI"]["ffmpeg_executable"] = ffmpeg_exe
        if latex_exe:
            _parser["CLI"]["latex_executable"] = latex_exe

        with open(library_wide, "w") as f:
            _parser.write(f)


def check_ffmpeg(ffmpeg_exe: str | None) -> None:
    ffmpeg_chk = check_ffmpeg_exe_working(ffmpeg_exe)
    ffmpeg_exe_temp, _ = prompt_user_for_choice(
        input_ffmpeg=(not ffmpeg_chk), input_latex=False
    )
    if ffmpeg_exe_temp:
        ffmpeg_exe = ffmpeg_exe_temp
    logger.debug(f"ffmpeg executable: {ffmpeg_exe}")
    if not config.ffmpeg_executable:
        save_config_latex_ffmpeg(ffmpeg_exe=ffmpeg_exe)
    config.ffmpeg_executable = ffmpeg_exe


def check_latex(latex_exe: str | None) -> None:
    latex_chk = check_latex_exe_working(latex_exe)
    _, latex_exe_temp = prompt_user_for_choice(
        input_ffmpeg=False, input_latex=(not latex_chk)
    )
    if latex_exe_temp:
        latex_exe = latex_exe_temp
    logger.debug(f"latex executable: {latex_exe}")
    if not config.latex_executable:
        save_config_latex_ffmpeg(latex_exe=latex_exe)
    config.latex_executable = latex_exe


def is_mp4_format() -> bool:
    """
    Determines if output format is .mp4

    Returns
    -------
    class:`bool`
        ``True`` if format is set as mp4

    """
    return config["format"] == "mp4"


def is_gif_format() -> bool:
    """
    Determines if output format is .gif

    Returns
    -------
    class:`bool`
        ``True`` if format is set as gif

    """
    return config["format"] == "gif"


def is_webm_format() -> bool:
    """
    Determines if output format is .webm

    Returns
    -------
    class:`bool`
        ``True`` if format is set as webm

    """
    return config["format"] == "webm"


def is_mov_format() -> bool:
    """
    Determines if output format is .mov

    Returns
    -------
    class:`bool`
        ``True`` if format is set as mov

    """
    return config["format"] == "mov"


def is_png_format() -> bool:
    """
    Determines if output format is .png

    Returns
    -------
    class:`bool`
        ``True`` if format is set as png

    """
    return config["format"] == "png"


def write_to_movie() -> bool:
    """
    Determines from config if the output is a video format such as mp4 or gif, if the --format is set as 'png'
    then it will take precedence event if the write_to_movie flag is set

    Returns
    -------
    class:`bool`
        ``True`` if the output should be written in a movie format

    """
    if is_png_format():
        return False
    return (
        config["write_to_movie"]
        or is_mp4_format()
        or is_gif_format()
        or is_webm_format()
        or is_mov_format()
    )


def add_extension_if_not_present(file_name: Path, extension: str) -> Path:
    if file_name.suffix != extension:
        return file_name.with_suffix(extension)
    else:
        return file_name


def add_version_before_extension(file_name: Path) -> Path:
    file_name = Path(file_name)
    path, name, suffix = file_name.parent, file_name.stem, file_name.suffix
    return Path(path, f"{name}_ManimCE_v{__version__}{suffix}")


def guarantee_existence(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=True)
    return path.resolve(strict=True)


def guarantee_empty_existence(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(str(path))
    path.mkdir(parents=True)
    return path.resolve(strict=True)


def seek_full_path_from_defaults(
    file_name: str, default_dir: Path, extensions: list[str]
) -> Path:
    possible_paths = [Path(file_name)]
    possible_paths += [
        Path(default_dir) / f"{file_name}{extension}" for extension in ["", *extensions]
    ]
    for path in possible_paths:
        if path.exists():
            return path
    error = f"From: {os.getcwd()}, could not find {file_name} at either of these locations: {possible_paths}"
    raise OSError(error)


def modify_atime(file_path) -> None:
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
        os.startfile(file_path if not in_browser else file_path.parent)
    else:
        if current_os == "Linux":
            commands = ["xdg-open"]
            file_path = file_path if not in_browser else file_path.parent
        elif current_os.startswith("CYGWIN"):
            commands = ["cygstart"]
            file_path = file_path if not in_browser else file_path.parent
        elif current_os == "Darwin":
            if is_gif_format():
                commands = ["ffplay", "-loglevel", config["ffmpeg_loglevel"].lower()]
            else:
                commands = ["open"] if not in_browser else ["open", "-R"]
        else:
            raise OSError("Unable to identify your operating system...")
        commands.append(file_path)
        sp.Popen(commands)


def open_media_file(file_writer: SceneFileWriter) -> None:
    file_paths = []

    if config["save_last_frame"]:
        file_paths.append(file_writer.image_file_path)
    if write_to_movie() and not is_gif_format():
        file_paths.append(file_writer.movie_file_path)
    if write_to_movie() and is_gif_format():
        file_paths.append(file_writer.gif_file_path)

    for file_path in file_paths:
        if config["show_in_file_browser"]:
            open_file(file_path, True)
        if config["preview"]:
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


def copy_template_files(
    project_dir: Path = Path("."), template_name: str = "Default"
) -> None:
    """Copies template files from templates dir to project_dir.

    Parameters
    ----------
        project_dir : :class:`Path`
            Path to project directory.
        template_name : :class:`str`
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
