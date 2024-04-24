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
        help="Preview the Scene's animation. OpenGL does a live preview in a "
        "popup window. Cairo opens the rendered video file in the system "
        "default media player.",
        default=None,
    ),
    option(
        "-f",
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
