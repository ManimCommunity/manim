"""building blocks of segmented video api"""

import enum
import typing


class SectionType(enum.Enum):
    """The type of a section defines how it is to be played in a presentation"""

    # start, end, wait for continuation by user
    normal = 1
    # start, end, immediately continue to next section
    skip = 2
    # start, end, restart, immediately continue to next section when continued by user
    loop = 3
    # start, end, restart, finish animation first when user continues
    complete_loop = 4


class Section:
    """a ::class `Scene` can be segmented into multiple Sections

    ...
    Attributes
    ----------
    type : SectionType
        how is this slide to be played, what is supposed to happen when it ends
    name : string
        human readable, not-unique name of this section
    """

    def __init__(self, type: SectionType, name: str, first_animation: int):
        self.type = type
        # names are not intended to be unique
        self.name = name
        # inclusive
        self.first_animation = first_animation
        # exclusive
        self.after_last_animation = first_animation
        self.video = ""

    def empty(self) -> bool:
        return self.first_animation == self.after_last_animation

    def set_video(self, video: str) -> None:
        self.video = video

    def get_dict(self) -> Dict:
        return {
            "slide_type": self.slide_type,
            "name": self.name,
            "slide_id": self.slide_id,
            "first_animation": self.first_animation,
            "after_last_animation": self.after_last_animation,
            "video": self.video,
        }

    def __repr__(self):
        return f"<Slide '{self.name}' from {self.first_animation} to {self.after_last_animation}, stored in '{self.video}'>"
