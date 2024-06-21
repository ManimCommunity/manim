from __future__ import annotations

from typing import TYPE_CHECKING

import moderngl_window as mglw
import numpy as np
from moderngl_window.context.pyglet.window import Window as FunWindow
from moderngl_window.timers.clock import Timer
from screeninfo import get_monitors

from .. import __version__, config


class Window(FunWindow):
    fullscreen: bool = False
    resizable: bool = False
    gl_version: tuple[int, int] = (3, 3)
    vsync: bool = True
    cursor: bool = True

    def __init__(self, size=config.window_size):
        # TODO: remove size argument from window init,
        # move size computation below to config

        monitors = get_monitors()
        mon_index = config.window_monitor
        monitor = monitors[min(mon_index, len(monitors) - 1)]

        if size == "default":
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
        else:
            size = tuple(size)

        super().__init__(size=size)
        self.pressed_keys = set()
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
        custom_position = config.window_position
        monitors = get_monitors()
        mon_index = config.window_monitor
        monitor = monitors[min(mon_index, len(monitors) - 1)]
        window_width, window_height = size
        # Position might be specified with a string of the form
        # x,y for integers x and y
        if "," in custom_position:
            return tuple(map(int, custom_position.split(",")))

        # Alternatively, it might be specified with a string like
        # UR, OO, DL, etc. specifying what corner it should go to
        char_to_n = {"L": 0, "U": 0, "O": 1, "R": 2, "D": 2}
        width_diff = monitor.width - window_width
        height_diff = monitor.height - window_height
        return (
            monitor.x + char_to_n[custom_position[1]] * width_diff // 2,
            -monitor.y + char_to_n[custom_position[0]] * height_diff // 2,
        )

    # Delegate event handling to scene
    def pixel_coords_to_space_coords(
        self, px: int, py: int, relative: bool = False
    ) -> np.ndarray:
        pw, ph = self.size
        # TODO
        fw, fh = (
            config.frame_width,
            config.frame_height,
        ) or self.scene.camera.get_frame_shape()
        fc = (
            config.frame_width,
            config.frame_height,
        ) or self.scene.camera.get_frame_center()
        if relative:
            return np.array([px / pw, py / ph, 0])
        else:
            return np.array(
                [fc[0] + px * fw / pw - fw / 2, fc[1] + py * fh / ph - fh / 2, 0]
            )
