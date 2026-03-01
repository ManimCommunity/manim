from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import moderngl_window as mglw
from moderngl_window.context.pyglet.window import Window as PygletWindow
from moderngl_window.timers.clock import Timer
from screeninfo import get_monitors

from manim import __version__, config
from manim.event_handler.window import WindowProtocol

if TYPE_CHECKING:
    from typing import TypeGuard

T = TypeVar("T")

if TYPE_CHECKING:
    pass

__all__ = ["Window"]


class Window(PygletWindow, WindowProtocol):
    name = "Manim Community"
    fullscreen: bool = False
    resizable: bool = False
    gl_version: tuple[int, int] = (3, 3)
    vsync: bool = True
    cursor: bool = True

    def __init__(self, window_size: str | tuple[int, ...] = config.window_size):
        # TODO: remove size argument from window init,
        # move size computation below to config

        monitors = get_monitors()
        mon_index = config.window_monitor
        monitor = monitors[min(mon_index, len(monitors) - 1)]

        invalid_window_size_error_message = (
            "window_size must be specified either as 'default', a string of the form "
            "'width,height', or a tuple of 2 ints of the form (width, height)."
        )

        if isinstance(window_size, tuple):
            if len(window_size) != 2:
                raise ValueError(invalid_window_size_error_message)
            size = window_size
        elif window_size == "default":
            # make window_width half the width of the monitor
            # but make it full screen if --fullscreen
            window_width = monitor.width
            if not config.fullscreen:
                window_width //= 2

            #  by default window_height = 9/16 * window_width
            window_height = int(
                window_width * config.frame_height // config.frame_width,
            )
            size = (window_width, window_height)
        elif len(window_size.split(",")) == 2:
            (window_width, window_height) = tuple(map(int, window_size.split(",")))
            size = (window_width, window_height)
        else:
            raise ValueError(invalid_window_size_error_message)

        super().__init__(size=size)
        self.pressed_keys: set = set()
        self.title = f"Manim Community {__version__}"
        self.size = size

        mglw.activate_context(window=self)
        self.timer = Timer()
        self.config = mglw.WindowConfig(ctx=self.ctx, wnd=self, timer=self.timer)
        self.timer.start()

        # No idea why, but when self.position is set once
        # it sometimes doesn't actually change the position
        # to the specified tuple on the rhs, but doing it
        # twice seems to make it work.  ¯\_(ツ)_/¯
        initial_position = self.find_initial_position(size)
        self.position = initial_position
        self.position = initial_position

    def find_initial_position(self, size: tuple[int, int]) -> tuple[int, int]:
        custom_position = config.window_position.replace(" ", "").upper()
        monitors = get_monitors()
        mon_index = config.window_monitor
        monitor = monitors[min(mon_index, len(monitors) - 1)]
        window_width, window_height = size

        # Position might be specified with a string of the form
        # x,y for integers x and y
        if "," in custom_position:
            pos = tuple(int(p) for p in custom_position.split(","))
            if tuple_len_2(pos):
                return pos
            else:
                raise ValueError("Expected position in the form x,y")

        # Alternatively, it might be specified with a string like
        # UR, OO, DL, etc. specifying what corner it should go to
        char_to_n = {"L": 0, "U": 0, "O": 1, "R": 2, "D": 2}
        width_diff: int = monitor.width - window_width
        height_diff: int = monitor.height - window_height

        return (
            monitor.x + char_to_n[custom_position[1]] * width_diff // 2,
            -monitor.y + char_to_n[custom_position[0]] * height_diff // 2,
        )


def tuple_len_2(pos: tuple[T, ...]) -> TypeGuard[tuple[T, T]]:
    return len(pos) == 2
