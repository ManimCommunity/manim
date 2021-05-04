import click
from cloup import option, option_group

output_options = option_group(
    "Output options",
    option(
        "-o",
        "--output_file",
        type=str,
        help="Specify the filename(s) of the rendered scene(s).",
    ),
    option(
        "--write_to_movie",
        is_flag=True,
        default=None,
        help="Write to a file.",
    ),
    option(
        "--media_dir",
        type=click.Path(),
        help="Path to store rendered videos and latex.",
    ),
    option("--log_dir", type=click.Path(), help="Path to store render logs."),
    option(
        "--log_to_file",
        is_flag=True,
        help="Log terminal output to file.",
    ),
)
