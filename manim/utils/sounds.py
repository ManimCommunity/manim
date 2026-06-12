"""Sound-related utility functions."""

from __future__ import annotations

__all__ = [
    "get_full_sound_file_path",
]

from typing import TYPE_CHECKING

from .. import config
from ..utils.file_ops import seek_full_path_from_defaults

if TYPE_CHECKING:
    from pathlib import Path

    from manim.typing import StrPath


# Still in use by add_sound() function in scene_file_writer.py
def get_full_sound_file_path(sound_file_name: StrPath) -> Path:
    return seek_full_path_from_defaults(
        sound_file_name,
        default_dir=config.get_dir("assets_dir"),
        extensions=[".wav", ".mp3"],
    )
