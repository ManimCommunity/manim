"""building blocks of segmented video api"""

import enum
from typing import Dict


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
    """a ::class `Scene` can be segmented into multiple Sections.
    It consists of multiple animations.

    ...
    Attributes
    ----------
    type : SectionType
        how is this slide to be played, what is supposed to happen when it ends
    name : string
        human readable, not-unique name of this section
    first_animation : int
        animation range start, inclusive
        gets set on creating of instance, doesn't change afterwards
    after_last_animation : int
        animation range end, exclusive
        gets increased every time a new animation is added
    video : str
        to be set once video for this section has been created
    """

    def __init__(self, type: SectionType, name: str, first_animation: int):
        self.type = type
        self.name = name
        self.first_animation = first_animation
        self.after_last_animation = first_animation
        self.video = ""

    def empty(self) -> bool:
        """are there no animations in this section?"""

        return self.first_animation == self.after_last_animation

    def get_dict(self) -> Dict:
        """get dictionary representation"""

        return {
            "type": self.type,
            "name": self.name,
            "first_animation": self.first_animation,
            "after_last_animation": self.after_last_animation,
            "video": self.video,
        }

    def __repr__(self):
        return f"<Section '{self.name}' from {self.first_animation} to {self.after_last_animation}, stored in '{self.video}'>"
