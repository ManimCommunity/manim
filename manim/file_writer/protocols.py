from __future__ import annotations

from typing import Protocol

from manim.typing import PixelArray


class FileWriterProtocol(Protocol):
    """Protocol for a file writer.

    This is mainly useful for testing purposes, to create
    a mock file writer. However, it can be used in plugins.
    """

    num_plays: int

    def __init__(self, scene_name: str) -> None: ...

    def begin_animation(self, allow_write: bool = False) -> object: ...

    def end_animation(self, allow_write: bool = False) -> object: ...

    def is_already_cached(self, hash_invocation: str) -> bool: ...

    def add_partial_movie_file(self, hash_animation: str) -> object: ...

    def write_frame(self, frame: PixelArray) -> object: ...

    def next_section(self, name: str, type_: str, skip_animations: bool) -> object: ...

    def finish(self) -> None: ...

    def save_image(self, image: PixelArray) -> object: ...
