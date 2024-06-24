from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable

    import manim.mobject.mobject as glmob
    from manim.event_handler.event_type import EventType


class EventListener:
    def __init__(
        self,
        mobject: glmob.OpenGLMobject,
        event_type: EventType,
        event_callback: Callable[[glmob.OpenGLMobject, dict[str, str]], None],
    ):
        self.mobject = mobject
        self.event_type = event_type
        self.callback = event_callback

    def __eq__(self, o: object) -> bool:
        return_val = False
        if isinstance(o, EventListener):
            try:
                return_val = (
                    self.callback == o.callback
                    and self.mobject == o.mobject
                    and self.event_type == o.event_type
                )
            except Exception:
                pass
        return return_val
