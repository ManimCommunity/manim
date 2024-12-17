from __future__ import annotations

import numpy as np
from typing_extensions import Any, Self

from manim.event_handler.event_listener import EventListener
from manim.event_handler.event_type import EventType


class EventDispatcher:
    def __init__(self) -> None:
        self.event_listeners: dict[EventType, list[EventListener]] = {
            event_type: [] for event_type in EventType
        }
        self.mouse_point = np.array((0.0, 0.0, 0.0))
        self.mouse_drag_point = np.array((0.0, 0.0, 0.0))
        self.pressed_keys: set[int] = set()
        self.draggable_object_listeners: list[EventListener] = []

    def add_listener(self, event_listener: EventListener) -> Self:
        assert isinstance(event_listener, EventListener)
        self.event_listeners[event_listener.event_type].append(event_listener)
        return self

    def remove_listener(self, event_listener: EventListener) -> Self:
        assert isinstance(event_listener, EventListener)
        try:
            while event_listener in self.event_listeners[event_listener.event_type]:
                self.event_listeners[event_listener.event_type].remove(event_listener)
        except Exception:
            # raise ValueError("Handler is not handling this event, so cannot remove it.")
            pass
        return self

    def dispatch(self, event_type: EventType, **event_data: Any) -> bool | None:
        if event_type == EventType.MouseMotionEvent:
            self.mouse_point = event_data["point"]
        elif event_type == EventType.MouseDragEvent:
            self.mouse_drag_point = event_data["point"]
        elif event_type == EventType.KeyPressEvent:
            self.pressed_keys.add(event_data["symbol"])  # Modifiers?
        elif event_type == EventType.KeyReleaseEvent:
            self.pressed_keys.difference_update({event_data["symbol"]})  # Modifiers?
        elif event_type == EventType.MousePressEvent:
            self.draggable_object_listeners = [
                listener
                for listener in self.event_listeners[EventType.MouseDragEvent]
                if listener.mobject.is_point_touching(self.mouse_point)
            ]
        elif event_type == EventType.MouseReleaseEvent:
            self.draggable_object_listeners = []

        propagate_event = None

        if event_type == EventType.MouseDragEvent:
            for listener in self.draggable_object_listeners:
                assert isinstance(listener, EventListener)
                propagate_event = listener.callback(listener.mobject, event_data)
                if propagate_event is not None and propagate_event is False:
                    return propagate_event

        elif event_type.value.startswith("mouse"):
            for listener in self.event_listeners[event_type]:
                if listener.mobject.is_point_touching(self.mouse_point):
                    propagate_event = listener.callback(listener.mobject, event_data)
                    if propagate_event is not None and propagate_event is False:
                        return propagate_event

        elif event_type.value.startswith("key"):
            for listener in self.event_listeners[event_type]:
                propagate_event = listener.callback(listener.mobject, event_data)
                if propagate_event is not None and propagate_event is False:
                    return propagate_event

        return propagate_event

    def get_listeners_count(self) -> int:
        return sum([len(value) for key, value in self.event_listeners.items()])

    def get_mouse_point(self) -> np.ndarray:
        return self.mouse_point

    def get_mouse_drag_point(self) -> np.ndarray:
        return self.mouse_drag_point

    def is_key_pressed(self, symbol: int) -> bool:
        return symbol in self.pressed_keys

    __iadd__ = add_listener
    __isub__ = remove_listener
    __call__ = dispatch
    __len__ = get_listeners_count
