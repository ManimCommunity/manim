"""Special rectangles."""

from __future__ import annotations

__all__ = [
    "ScreenRectangle",
    "FullScreenRectangle",
]


from typing import Any

from manim.mobject.geometry.polygram import Rectangle

from .. import config


class ScreenRectangle(Rectangle):
    def __init__(
        self, aspect_ratio: float = 16.0 / 9.0, height: float = 4, **kwargs: Any
    ) -> None:
        super().__init__(width=aspect_ratio * height, height=height, **kwargs)

    @property
    def aspect_ratio(self) -> float:
        """The aspect ratio.

        When set, the width is stretched to accommodate
        the new aspect ratio.
        """
        return self.width / self.height

    @aspect_ratio.setter
    def aspect_ratio(self, value: float) -> None:
        self.stretch_to_fit_width(value * self.height)


class FullScreenRectangle(ScreenRectangle):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.height = config["frame_height"]
