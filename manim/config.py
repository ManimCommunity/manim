"""
config.py
---------
Process the manim.cfg file and the command line arguments into a single
config object.
"""
import os
import sys
from contextlib import contextmanager

import colour

from . import constants
from .utils.config_utils import _run_config, _init_dirs, _from_command_line

from .logger import logger
from .utils.tex import TexTemplate, TexTemplateFromFile

__all__ = ["file_writer_config", "config", "camera_config", "tempconfig"]


config = None


@contextmanager
def tempconfig(temp):
    """Context manager that temporarily modifies the global config dict.

    The code block inside the ``with`` statement will use the modified config.
    After the code block, the config will be restored to its original value.

    Parameters
    ----------

    temp : :class:`dict`
        A dictionary whose keys will be used to temporarily update the global
        config.

    Examples
    --------
    Use ``with tempconfig({...})`` to temporarily change the default values of
    certain objects.

    .. code_block:: python

       c = Camera()
       c.frame_width == config['frame_width']        # -> True
       with tempconfig({'frame_width': 100}):
           c = Camera()
           c.frame_width == config['frame_width']    # -> False
           c.frame_width == 100                      # -> True

    """
    global config
    original = config.copy()

    temp = {k: v for k, v in temp.items() if k in original}

    # In order to change the config that every module has acces to, use
    # update(), DO NOT use assignment.  Assigning config = some_dict will just
    # make the local variable named config point to a new dictionary, it will
    # NOT change the dictionary that every module has a reference to.
    config.update(temp)
    try:
        yield
    finally:
        config.update(original)  # update, not assignment!


def _parse_config(config_parser, args):
    """Parse config files and CLI arguments into a single dictionary."""
    # By default, use the CLI section of the digested .cfg files
    default = config_parser["CLI"]

    # Handle the *_quality flags.  These determine the section to read
    # and are stored in 'camera_config'.  Note the highest resolution
    # passed as argument will be used.
    for flag in ["fourk_quality", "high_quality", "medium_quality", "low_quality"]:
        if getattr(args, flag):
            section = config_parser[flag]
            break
    else:
        section = config_parser["CLI"]
    config = {opt: section.getint(opt) for opt in config_parser[flag]}

    config["default_pixel_height"] = default.getint("pixel_height")
    config["default_pixel_width"] = default.getint("pixel_width")
    # The -r, --resolution flag overrides the *_quality flags
    if args.resolution is not None:
        if "," in args.resolution:
            height_str, width_str = args.resolution.split(",")
            height, width = int(height_str), int(width_str)
        else:
            height, width = int(args.resolution), int(16 * height / 9)
        config.update({"pixel_height": height, "pixel_width": width})

    # Handle the -c (--background_color) flag
    if args.background_color is not None:
        try:
            background_color = colour.Color(args.background_color)
        except AttributeError as err:
            logger.warning("Please use a valid color.")
            logger.error(err)
            sys.exit(2)
    else:
        background_color = colour.Color(default["background_color"])
    config["background_color"] = background_color

    # Set the rest of the frame properties
    config["frame_height"] = 8.0
    config["frame_width"] = (
        config["frame_height"] * config["pixel_width"] / config["pixel_height"]
    )
    config["frame_y_radius"] = config["frame_height"] / 2
    config["frame_x_radius"] = config["frame_width"] / 2
    config["top"] = config["frame_y_radius"] * constants.UP
    config["bottom"] = config["frame_y_radius"] * constants.DOWN
    config["left_side"] = config["frame_x_radius"] * constants.LEFT
    config["right_side"] = config["frame_x_radius"] * constants.RIGHT

    # Handle the --tex_template flag.  Note we accept None if the flag is absent
    tex_fn = os.path.expanduser(args.tex_template) if args.tex_template else None

    if tex_fn is not None and not os.access(tex_fn, os.R_OK):
        # custom template not available, fallback to default
        logger.warning(
            f"Custom TeX template {tex_fn} not found or not readable. "
            "Falling back to the default template."
        )
        tex_fn = None
    config["tex_template_file"] = tex_fn
    config["tex_template"] = (
        TexTemplateFromFile(filename=tex_fn) if tex_fn is not None else TexTemplate()
    )

    return config


args, config_parser, file_writer_config, successfully_read_files = _run_config()
logger.setLevel(file_writer_config["verbosity"])
if _from_command_line():
    logger.debug(
        f"Read configuration files: {[os.path.abspath(cfgfile) for cfgfile in successfully_read_files]}"
    )
    if not (hasattr(args, "subcommands")):
        _init_dirs(file_writer_config)
config = _parse_config(config_parser, args)
camera_config = config
