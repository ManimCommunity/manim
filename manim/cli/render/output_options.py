from __future__ import annotations

from cloup import IntRange, Path, option, option_group

__all__ = ["output_options"]

output_options = option_group(
    "Output options",
    option(
        "-o",
        "--output_file",
        type=str,
        default=None,
        help="Specify the filename(s) of the rendered scene(s).",
    ),
    option(
        "-0",
        "--zero_pad",
        type=IntRange(0, 9),
        default=None,
        help="Zero padding for PNG file names.",
    ),
    option(
        "--write_to_movie",
        is_flag=True,
        default=None,
        help="Write the video rendered with opengl to a file.",
    ),
    option(
        "--media_dir",
        type=Path(),
        default=None,
        help="Path to store rendered videos and latex.",
    ),
    option(
        "--log_dir",
        type=Path(),
        help="Path to store render logs.",
        default=None,
    ),
    option(
        "--log_to_file",
        is_flag=True,
        default=None,
        help="Log terminal output to file.",
    ),
)
