"""building blocks of segmented video API"""

import os
from enum import Enum
from typing import Any, Dict, List, Optional

from manim import get_video_metadata


class DefaultSectionType(str, Enum):
    """The type of a section can be used for third party applications.
    A presentation system could for example use the types to created loops.

    Examples
    --------
    This class can be reimplemented for more types:
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
    """a ::class `Scene` can be segmented into multiple Sections.
    It consists of multiple animations.

    ...
    Attributes
    ----------
    type : SectionType
    name : string
        human readable, not-unique name of this section
    partial_movie_files : list of strings or Nones
        animations belonging to this section
        None when not to be saved -> still keeps section alive
    video : str or None
        path of video file from sections directory with animations belonging to section
        None -> section is not to be saved
    """

    def __init__(self, type: str, video: Optional[str], name: str):
        self.type = type
        self.video: Optional[str] = video
        self.name = name
        self.partial_movie_files: List[Optional[str]] = []

    def empty(self) -> bool:
        """Are there no animations in this section?
        None animations also get counted.
        """

        return len(self.partial_movie_files) == 0

    def get_clean_partial_movie_files(self) -> List[str]:
        """return not None partial_movie_files"""
        return [el for el in self.partial_movie_files if el is not None]

    def get_dict(self, sections_dir: str) -> Dict[str, Any]:
        """Get dictionary representation with metadata of output video.
        The output from this function is used from every section to build the sections index file.
        The output video must have been created in the `sections_dir` before executing this method.
        This is the main part of the Segmented Video API.
        """
        if self.video is None:
            raise ValueError(
                f"section '{self.name} can't be exported as dict, it doesn't have a video path assigned to it'"
            )

        video_metadata = get_video_metadata(os.path.join(sections_dir, self.video))
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
