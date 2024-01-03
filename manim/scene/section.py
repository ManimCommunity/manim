"""building blocks of segmented video API"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from manim import get_video_metadata

__all__ = ["Section", "DefaultSectionType"]


class DefaultSectionType(str, Enum):
    """The type of a section can be used for third party applications.
    A presentation system could for example use the types to created loops.

    Examples
    --------
    This class can be reimplemented for more types::

        class PresentationSectionType(str, Enum):
            # start, end, wait for continuation by user
            NORMAL = "presentation.normal"
            # start, end, immediately continue to next section
            SKIP = "presentation.skip"
            # start, end, restart, immediately continue to next section when continued by user
            LOOP = "presentation.loop"
            # start, end, restart, finish animation first when user continues
            COMPLETE_LOOP = "presentation.complete_loop"
    """

    NORMAL = "default.normal"


class Section:
    """A :class:`.Scene` can be segmented into multiple Sections.
    Refer to :doc:`the documentation</tutorials/output_and_config>` for more info.
    It consists of multiple animations.

    Attributes
    ----------
    type
        Can be used by a third party applications to classify different types of sections.
    video
        Path to video file with animations belonging to section relative to sections directory.
        If ``None``, then the section will not be saved.
    name
        Human readable, non-unique name for this section.
    skip_animations
        Skip rendering the animations in this section when ``True``.
    partial_movie_files
        Animations belonging to this section.

    See Also
    --------
    :class:`.DefaultSectionType`
    :meth:`.CairoRenderer.update_skipping_status`
    :meth:`.OpenGLRenderer.update_skipping_status`
    """

    def __init__(self, type: str, video: str | None, name: str, skip_animations: bool):
        self.type = type
        # None when not to be saved -> still keeps section alive
        self.video: str | None = video
        self.name = name
        self.skip_animations = skip_animations
        self.partial_movie_files: list[str | None] = []

    def is_empty(self) -> bool:
        """Check whether this section is empty.

        Note that animations represented by ``None`` are also counted.
        """
        return len(self.partial_movie_files) == 0

    def get_clean_partial_movie_files(self) -> list[str]:
        """Return all partial movie files that are not ``None``."""
        return [el for el in self.partial_movie_files if el is not None]

    def get_dict(self, sections_dir: Path) -> dict[str, Any]:
        """Get dictionary representation with metadata of output video.

        The output from this function is used from every section to build the sections index file.
        The output video must have been created in the ``sections_dir`` before executing this method.
        This is the main part of the Segmented Video API.
        """
        if self.video is None:
            raise ValueError(
                f"Section '{self.name}' cannot be exported as dict, it does not have a video path assigned to it"
            )

        video_metadata = get_video_metadata(sections_dir / self.video)
        return dict(
            {
                "name": self.name,
                "type": self.type,
                "video": self.video,
            },
            **video_metadata,
        )

    def __repr__(self):
        return f"<Section '{self.name}' stored in '{self.video}'>"
