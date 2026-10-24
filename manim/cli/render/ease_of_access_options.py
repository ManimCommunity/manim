from __future__ import annotations

from cloup import Choice, option, option_group

__all__ = ["ease_of_access_options"]

ease_of_access_options = option_group(
    "Ease of access options",
    option(
        "--progress_bar",
        default=None,
        show_default=False,
        type=Choice(
            ["display", "leave", "none"],
            case_sensitive=False,
        ),
        help="Display progress bars and/or keep them displayed.",
    ),
    option(
        "-p",
        "--preview",
        is_flag=True,
        help="Open the completed media artifact after rendering.",
        default=None,
    ),
    option(
        "-l",
        "--live-preview",
        is_flag=True,
        help="Display frames in a renderer-provided live preview. With "
        "--format=auto, no media file is written.",
        default=None,
    ),
    option(
        "--show_in_file_browser",
        is_flag=True,
        help="Show the output file in the file browser.",
        default=None,
    ),
    option(
        "--jupyter",
        is_flag=True,
        help="Using jupyter notebook magic.",
        default=None,
    ),
)
