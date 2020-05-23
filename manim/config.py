import argparse
import colour
import os
import sys
import types

from . import constants
from . import dirs
from .logging import logger

def parse_cli():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "file",
            help="path to file holding the python code for the scene",
        )
        parser.add_argument(
            "scene_names",
            nargs="*",
            help="Name of the Scene class you want to see",
        )
        parser.add_argument(
            "-p", "--preview",
            action="store_true",
            help="Automatically open the saved file once its done",
        )
        parser.add_argument(
            "-w", "--write_to_movie",
            action="store_true",
            help="Render the scene as a movie file",
        )
        parser.add_argument(
            "-s", "--save_last_frame",
            action="store_true",
            help="Save the last frame",
        )
        parser.add_argument(
            "--dry_run", 
            action="store_true",
            help= "Do a dry run (render scenes but generate no output files)",
        )
        parser.add_argument(
            "-l", "--low_quality",
            action="store_true",
            help="Render at low quality (for fastest rendering)",
        ),
        parser.add_argument(
            "-m", "--medium_quality",
            action="store_true",
            help="Render at medium quality (for much faster rendering)",
        ),
        parser.add_argument(
            "-e", "--high_quality",
            action="store_true",
            help="Render at high quality (for slightly faster rendering)",
        ),
        parser.add_argument(
            "-k", "--four_k",
            action="store_true",
            help="Render at 4K quality (slower rendering)",
        ),
        parser.add_argument(
            "-g", "--save_pngs",
            action="store_true",
            help="Save each frame as a png",
        )
        parser.add_argument(
            "-i", "--save_as_gif",
            action="store_true",
            help="Save the video as gif",
        )
        parser.add_argument(
            "-f", "--show_file_in_finder",
            action="store_true",
            help="Show the output file in finder",
        )
        parser.add_argument(
            "-t", "--transparent",
            action="store_true",
            help="Render to a movie file with an alpha channel",
        )
        parser.add_argument(
            "-q", "--quiet",
            action="store_true",
            help="",
        )
        parser.add_argument(
            "-a", "--write_all",
            action="store_true",
            help="Write all the scenes from a file",
        )
        parser.add_argument(
            "-o", "--file_name",
            help="Specify the name of the output file, if"
                 "it should be different from the scene class name",
        )
        parser.add_argument(
            "-n", "--start_at_animation_number",
            help="Start rendering not from the first animation, but"
                 "from another, specified by its index.  If you pass"
                 "in two comma separated values, e.g. \"3,6\", it will end"
                 "the rendering at the second value",
        )
        parser.add_argument(
            "-r", "--resolution",
            help="Resolution, passed as \"height,width\"",
        )
        parser.add_argument(
            "-c", "--color",
            help="Background color",
        )
        parser.add_argument(
            "--sound",
            action="store_true",
            help="Play a success/failure sound",
        )
        parser.add_argument(
            "--leave_progress_bars",
            action="store_true",
            help="Leave progress bars displayed in terminal",
        )
        parser.add_argument(
            "--media_dir",
            help="directory to write media",
        )
        video_group = parser.add_mutually_exclusive_group()
        video_group.add_argument(
            "--video_dir",
            help="directory to write file tree for video",
        )
        parser.add_argument(
            "--tex_dir",
            help="directory to write tex",
        )
        parser.add_argument(
            "--text_dir",
            help="directory to write text",
        )
        return parser.parse_args()
    except argparse.ArgumentError as err:
        logger.error(str(err))
        sys.exit(2)


def get_configuration(args):
    file_writer_config = {
        # By default, write to file
        "write_to_movie": (args.write_to_movie or not args.save_last_frame) and not args.dry_run,
        "save_last_frame": args.save_last_frame and not args.dry_run,
        "save_pngs": args.save_pngs,
        "save_as_gif": args.save_as_gif,
        # If -t is passed in (for transparent), this will be RGBA
        "png_mode": "RGBA" if args.transparent else "RGB",
        "movie_file_extension": ".mov" if args.transparent else ".mp4",
        "file_name": args.file_name,
        "input_file_path": args.file,
    }
    config = {
        "file": args.file,
        "scene_names": args.scene_names,
        "open_video_upon_completion": args.preview,
        "show_file_in_finder": args.show_file_in_finder,
        "file_writer_config": file_writer_config,
        "quiet": args.quiet or args.write_all,
        "ignore_waits": args.preview,
        "write_all": args.write_all,
        "start_at_animation_number": args.start_at_animation_number,
        "end_at_animation_number": None,
        "sound": args.sound,
        "leave_progress_bars": args.leave_progress_bars,
        "media_dir": args.media_dir,
        "video_dir": args.video_dir,
        "tex_dir": args.tex_dir,
        "text_dir": args.text_dir,
    }

    # Camera configuration
    config["camera_config"] = get_camera_configuration(args)

    # Arguments related to skipping
    stan = config["start_at_animation_number"]
    if stan is not None:
        if "," in stan:
            start, end = stan.split(",")
            config["start_at_animation_number"] = int(start)
            config["end_at_animation_number"] = int(end)
        else:
            config["start_at_animation_number"] = int(stan)

    config["skip_animations"] = any([
        file_writer_config["save_last_frame"],
        config["start_at_animation_number"],
    ])
    return config


def get_camera_configuration(args):
    camera_config = {}
    if args.low_quality:
        camera_config.update(constants.LOW_QUALITY_CAMERA_CONFIG)
    elif args.medium_quality:
        camera_config.update(constants.MEDIUM_QUALITY_CAMERA_CONFIG)
    elif args.high_quality:
        camera_config.update(constants.HIGH_QUALITY_CAMERA_CONFIG)
    elif args.four_k:
        camera_config.update(constants.FOURK_CAMERA_CONFIG)
    else:
        camera_config.update(constants.PRODUCTION_QUALITY_CAMERA_CONFIG)

    # If the resolution was passed in via -r
    if args.resolution:
        if  "," in args.resolution:
            height_str, width_str = args.resolution.split(",")
            height = int(height_str)
            width = int(width_str)
        else:
            height = int(args.resolution)
            width = int(16 * height / 9)
        camera_config.update({
            "pixel_height": height,
            "pixel_width": width,
        })

    if args.color:
        try:
            camera_config["background_color"] = colour.Color(args.color)
        except AttributeError as err:
            logger.warning("Please use a valid color")
            logger.error(err)
            sys.exit(2)

    # If rendering a transparent image/move, make sure the
    # scene has a background opacity of 0
    if args.transparent:
        camera_config["background_opacity"] = 0

    return camera_config


def initialize_directories(config):
    dir_config = {}
    dir_config["media_dir"] = config["media_dir"] or dirs.MEDIA_DIR
    dir_config["video_dir"] = config["video_dir"] or dirs.VIDEO_DIR

    if not (config["video_dir"] and config["tex_dir"]):
        if config["media_dir"]:
            if not os.path.isdir(dir_config["media_dir"]):
                os.makedirs(dir_config["media_dir"])
        if not os.path.isdir(dir_config["media_dir"]):
            dir_config["media_dir"] = "./media"
        else:
            print(
                f"Media will be written to {dir_config['media_dir'] + os.sep}. You can change "
                "this behavior with the --media_dir flag, or by adjusting dirs.py.,"
            )
    else:
        if config["media_dir"]:
            print(
                "Ignoring --media_dir, since both --tex_dir and --video_dir were passed."
            )

    dir_config["tex_dir"] = (config["tex_dir"]
                             or dirs.TEX_DIR
                             or os.path.join(dir_config["media_dir"], "Tex"))
    dir_config["text_dir"] = (config["text_dir"]
                              or dirs.TEXT_DIR
                              or os.path.join(dir_config["media_dir"], "texts"))

    if not config["video_dir"] or dirs.VIDEO_DIR:
        dir_config["video_dir"] = os.path.join(dir_config["media_dir"], "videos")
        
    for folder in [dir_config["video_dir"], dir_config["tex_dir"], dir_config["text_dir"]]:
        if folder != "" and not os.path.exists(folder):
            os.makedirs(folder)

    dirs.MEDIA_DIR = dir_config["media_dir"]
    dirs.VIDEO_DIR = dir_config["video_dir"]
    dirs.TEX_DIR = dir_config["tex_dir"]
    dirs.TEXT_DIR = dir_config["text_dir"]
