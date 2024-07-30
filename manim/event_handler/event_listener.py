from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from typing_extensions import Any

    from manim.event_handler.event_type import EventType
    from manim.mobject.opengl.opengl_mobject import OpenGLMobject


class EventListener:
    def __init__(
        self,
        mobject: OpenGLMobject,
        event_type: EventType,
        event_callback: Callable[[OpenGLMobject, dict[str, str]], None],
    ) -> None:
        self.mobject = mobject
        self.event_type = event_type
        self.callback = event_callback

    def __eq__(self, other: Any) -> bool:
        return_val = False
        if isinstance(other, EventListener):
            with contextlib.suppress(Exception):
                return_val = (
                    self.callback == other.callback
                    and self.mobject == other.mobject
                    and self.event_type == other.event_type
                )
        return return_val
