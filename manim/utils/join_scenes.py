from pathlib import Path
import subprocess

from manim._config import config, tempconfig, logger
from manim.constants import QUALITIES
from manim.scene.scene import Scene
from manim.__main__ import __version__

__all__ = ["render_multiple_scenes"]

def combine_renders(
        dir: Path,
        input_files: list[str],
        output_file: Path,
    ) -> None:
    """Make ffmpeg do all the hard work of combining video files"""

    file_list = dir / "partial_scenes_file_list.txt"
    logger.debug(
        f"Renders to combine ({len(input_files)} files): %(p)s",
        {"p": input_files[:5]},
    )
    with file_list.open("w", encoding="utf-8") as fp:
        fp.write("# This file is used internally by FFMPEG.\n")
        for pf_path in input_files:
            pf_path = Path(pf_path).as_posix()
            fp.write(f"file 'file:{pf_path}'\n")
    commands = [
        config.ffmpeg_executable,
        "-y",  # overwrite output file if it exists
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(file_list),
        "-loglevel",
        config.ffmpeg_loglevel.lower(),
        "-metadata",
        f"comment=Rendered with Manim Community v{__version__}",
        "-nostdin",
        "-c",
        "copy",
        str(output_file)
    ]

    combine_process = subprocess.Popen(commands)
    combine_process.wait()

def combine_scenes(
        *classes: type[Scene],
        output: Path | None = None
    ) -> Path:
    """
    Combine scenes produced by different renders. This function is mostly Path stuff.

    Returns:
    --------
        The location of the final file
    """
    quality_dict = QUALITIES[config.quality] # type: ignore
    quality_folder = f"{quality_dict['pixel_height']}p{quality_dict['frame_rate']}"

    dir = Path(config.media_dir) / f"videos/{quality_folder}"

    files = [
        str(dir / f"{cls.__name__}.mp4")
        for cls in classes
    ]

    output = dir/"final.mp4" if output is None else output

    combine_renders(
        dir,
        files,
        output
    )
    return output

def render_multiple_scenes(
        *scenes: type[Scene],
        dir: Path | str = config.media_dir,
        user_config: dict = {},
        output_file: Path | str | None = None
    ) -> None:
    """
    Render and concatenate multiple `Scene`s together.

    Parameters:
    -----------
        *scenes
            The classes to combine
        dir
            The location of the media files
        user_config
            Other configuration options, like transparency and stuff. Some features (like turning into a GIF) may not work
        output_file
            Where to keep the final file
    """
    if not scenes:
        raise ValueError("Must be at least one scene to render!")

    if not Path(dir).exists():
        Path(dir).mkdir()
    config.media_dir = str(dir)

    config.update(user_config)

    for scene in scenes:
        # use tempconfig to prevent bugs
        # with multiple renders in one file
        with tempconfig({}):
            scene().render()
    
    if output_file is not None:
        output_file = Path(output_file)
    final = combine_scenes(*scenes, output=output_file)
    
    logger.info(f"Concatenated Scene can be found at \'{final.absolute()}\'")


