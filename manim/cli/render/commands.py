from inspect import Traceback
import os
import re
import sys
import click

from manim import config, logger

from manim.constants import EPILOG
from manim.constants import CONTEXT_SETTINGS
from manim.utils.module_ops import scene_classes_from_file
from manim.utils.file_ops import open_file as open_media_file

from pathlib import Path

from click_option_group import optgroup


def validate_scene_range(ctx, param, value):
    try:
        start = int(value)
        return (start,)
    except:
        pass

    if value:
        try:
            start, end = map(int, re.split(";|,|-", value))
            return (
                start,
                end,
            )
        except:
            logger.error("Couldn't determine a range for -n option.")
            exit()


def validate_resolution(ctx, param, value):
    if value:
        try:
            start, end = map(int, re.split(";|,|-", value))
            return (
                start,
                end,
            )
        except:
            logger.error("Resolution option is invalid.")
            exit()


def open_file_if_needed(file_writer):
    if config["verbosity"] != "DEBUG":
        curr_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    open_file = any([config["preview"], config["show_in_file_browser"]])

    if open_file:
        file_paths = []

        if config["save_last_frame"]:
            file_paths.append(file_writer.image_file_path)
        if config["write_to_movie"] and not config["save_as_gif"]:
            file_paths.append(file_writer.movie_file_path)
        if config["save_as_gif"]:
            file_paths.append(file_writer.gif_file_path)

        for file_path in file_paths:
            if config["show_in_file_browser"]:
                open_media_file(file_path, True)
            if config["preview"]:
                open_media_file(file_path, False)

    if config["verbosity"] != "DEBUG":
        sys.stdout.close()
        sys.stdout = curr_stdout


@click.group(
    invoke_without_command=True,
    context_settings=CONTEXT_SETTINGS,
    epilog=EPILOG,
)
@click.argument("file", required=False)
@click.argument("scenes", required=False, nargs=-1)
@optgroup.group("Global options")
@optgroup.option(
    "-c",
    "--config_file",
    help="Specify the configuration file to use for render settings.",
)
@optgroup.option(
    "--custom_folders",
    is_flag=True,
    help="Use the folders defined in the [custom_folders] section of the config file to define the output folder structure.",
)
@optgroup.option(
    "--disable_caching",
    is_flag=True,
    help="Disable the use of the cache (still generates cache files).",
)
@optgroup.option(
    "--flush_cache", is_flag=True, help="Remove cached partial movie files."
)
@optgroup.option("--tex_template", help="Specify a custom TeX template file.")
@optgroup.option(
    "-v",
    "--verbose",
    type=click.Choice(
        [
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ],
        case_sensitive=False,
    ),
    help=" Verbosity of CLI output. Changes ffmpeg log level unless 5+.",
)
@optgroup.group("Output options")
@optgroup.option(
    "-o",
    "--output",
    multiple=True,
    help="Specify the filename(s) of the rendered scene(s).",
)
@optgroup.option(
    "--media_dir", type=click.Path(), help="Path to store rendered videos and latex."
)
@optgroup.option("--log_dir", type=click.Path(), help="Path to store render logs.")
@optgroup.option(
    "--log_to_file",
    default=True,
    show_default=True,
    is_flag=True,
    help="Log terminal output to file",
)
@optgroup.group("Render Options")
@optgroup.option(
    "-n",
    "--from_animation_number",
    callback=validate_scene_range,
    help="Start rendering from n_0 until n_1. If n_1 is left unspecified, renders all scenes after n_0.",
)
@optgroup.option(
    "-a",
    "--write_all",
    help="Render all scenes in the input file.",
)
@optgroup.option(
    "-f",
    "--format",
    default="mp4",
    type=click.Choice(
        [
            "png",
            "gif",
            "mp4",
        ],
        case_sensitive=False,
    ),
)
@optgroup.option("-s", "--save_last_frame", is_flag=True)
@optgroup.option(
    "-q",
    "--quality",
    default="h",
    type=click.Choice(
        [
            "l",
            "m",
            "h",
            "p",
            "k",
        ],
        case_sensitive=False,
    ),
    help="""
    Render quality at the follow resolution framerates, respectively:
    854x480 30FPS, 
    1280x720 30FPS,
    1920x1080 60FPS,
    2560x1440 60FPS,
    3840x2160 60FPS
    """,
)
@optgroup.option(
    "-r",
    "--resolution",
    callback=validate_resolution,
    help="Resolution in (W,H) for when 16:9 aspect ratio isn't possible.",
)
@optgroup.option(
    "--fps",
    default=30,
    show_default=True,
    type=float,
    help="Render at this frame rate.",
)
@optgroup.option(
    "--webgl_renderer",
    # show_default=f"Current directory {os.getcwd()}",
    default=None,
    type=click.Path(),
    help="Render scenes using the WebGL frontend. Requires a path to the WebGL frontend.",
)
@optgroup.option(
    "-t", "--transparent", is_flag=True, help="Render scenes with alpha channel."
)
@optgroup.option(
    "-c",
    "--background_color",
    show_default=True,
    default="#000000",
    help="Render scenes with background color.",
)
@optgroup.group("Ease of access options")
@optgroup.option(
    "--progress_bar",
    default="display",
    show_default=True,
    type=click.Choice(
        [
            "display",
            "leave",
            "none",
        ],
        case_sensitive=False,
    ),
    help="Display progress bars and/or keep them displayed.",
)
@optgroup.option(
    "-p",
    "--preview",
    is_flag=True,
    help="Preview the rendered file(s) in default player.",
)
@optgroup.option(
    "-f",
    "--show_in_file_browser",
    is_flag=True,
    help="Show the output file in the file browser.",
)
@optgroup.option("--sound", is_flag=True, help="Play a success/failure sound.")
@optgroup.option("--jupyter", is_flag=True, help="Using jupyter notebook magic.")
@click.pass_context
def render(
    ctx,
    file,
    scenes,
    config_file,
    custom_folders,
    disable_caching,
    flush_cache,
    tex_template,
    verbose,
    output,
    media_dir,
    log_dir,
    log_to_file,
    from_animation_number,
    write_all,
    format,
    save_last_frame,
    quality,
    resolution,
    fps,
    webgl_renderer,
    transparent,
    background_color,
    progress_bar,
    preview,
    show_in_file_browser,
    sound,
    jupyter,
):
    """Render SCENE(S) from the input FILE.

    FILE is the file path of the script.

    SCENES is an optional list of scenes in the file.
    """
    args = {
        "ctx": ctx,
        "file": file,
        "scene_names": scenes,
        "config_file": config_file,
        "custom_folders": custom_folders,
        "disable_caching": disable_caching,
        "flush_cache": flush_cache,
        "tex_template": tex_template,
        "verbosity": verbose,
        "output_file": output,
        "media_dir": media_dir,
        "log_dir": log_dir,
        "log_to_file": log_to_file,
        "from_animation_number": from_animation_number,
        "write_all": write_all,
        "format": format,
        "save_last_frame": save_last_frame,
        "quality": quality,
        "resolution": resolution,
        "frame_rate": fps,
        "webgl_renderer": webgl_renderer,
        "transparent": transparent,
        "background_color": background_color,
        "progress_bar": progress_bar,
        "preview": preview,
        "show_in_file_browser": show_in_file_browser,
        "sound": sound,
        "jupyter": jupyter,
    }

    class ClickArgs:
        def __init__(self, args):
            for name in args:
                setattr(self, name, args[name])

        def _get_kwargs(self):
            return list(self.__dict__.items())

        def __eq__(self, other):
            if not isinstance(other, ClickArgs):
                return NotImplemented
            return vars(self) == vars(other)

        def __contains__(self, key):
            return key in self.__dict__

        def __repr__(self):
            return str(self.__dict__)

    click_args = ClickArgs(args)
    if jupyter:
        return click_args
    config.digest_args(click_args)

    if webgl_renderer:
        try:
            from manim.grpc.impl import frame_server_impl

            server = frame_server_impl.get(file)
            server.start()
            server.wait_for_termination()
        except ModuleNotFoundError as e:
            print("\n\n")
            print(
                "Dependencies for the WebGL render are missing. Run "
                "pip install manim[webgl_renderer] to install them."
            )
            print(e)
            print("\n\n")
    else:
        for SceneClass in scene_classes_from_file(Path(file)):
            try:
                scene = SceneClass()
                scene.render()
                open_file_if_needed(scene.renderer.file_writer)
            except Exception as e:
                print(f"Exception {e}")
    return args
