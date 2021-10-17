"""building blocks of segmented video api"""

from enum import Enum
from typing import Dict, List, Optional


class DefaultSectionType(str, Enum):
    """The type of a section defines how it is to be played in a presentation.
    This class can be reimplemented for more types:

    Examples
    --------
    >>> class PresentationSectionType(str, Enum):
    >>>     # start, end, wait for continuation by user
    >>>     NORMAL = "presentation.normal"
    >>>     # start, end, immediately continue to next section
    >>>     SKIP = "presentation.skip"
    >>>     # start, end, restart, immediately continue to next section when continued by user
    >>>     LOOP = "presentation.loop"
    >>>     # start, end, restart, finish animation first when user continues
    >>>     COMPLETE_LOOP = "presentation.complete_loop"
    """

    NORMAL = "default.normal"


class Section:
    """a ::class `Scene` can be segmented into multiple Sections.
    It consists of multiple animations.

    ...
    Attributes
    ----------
    type : SectionType
        how is this slide to be played, what is supposed to happen when it ends
    name : string
        human readable, not-unique name of this section
    partial_movie_files : list of strings or Nones
        animations belonging to this section
        None when not to be rendered -> still keeps section alive
    video : str or None
        path of video file with animations belonging to section
        None -> section is not to be saved
    """

    def __init__(self, type: str, video: Optional[str], name: str):
        self.type = type
        self.video: Optional[str] = video
        self.name = name
        self.partial_movie_files: List[Optional[str]] = []

    def empty(self) -> bool:
        """are there no animations in this section?"""

        return len(self.partial_movie_files) == 0

    def get_clean_partial_movie_files(self) -> List[str]:
        """return not None partial_movie_files"""
        return [el for el in self.partial_movie_files if el is not None]

    def get_dict(self) -> Dict[str, str]:
        """get dictionary representation"""
        if self.video is None:
            raise ValueError(
                f"section '{self.name} can't be exported as dict, it doesn't have a video path assigned to it'"
            )

        return {
            "name": self.name,
            "type": self.type,
            "video": self.video,
        }

    def __repr__(self):
        return f"<Section '{self.name}' stored in '{self.video}'>"
