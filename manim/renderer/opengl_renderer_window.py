from __future__ import annotations

from typing import TYPE_CHECKING

import moderngl_window as mglw
import numpy as np
from moderngl_window.context.pyglet.window import Window as PygletWindow
from moderngl_window.timers.clock import Timer
from screeninfo import get_monitors

from .. import __version__, config

if TYPE_CHECKING:
    import manim.scene as m_scene


class Window(PygletWindow):
    fullscreen: bool = False
    resizable: bool = True
    gl_version: tuple[int, int] = (3, 3)
    vsync: bool = True
    cursor: bool = True

    def __init__(self, scene: m_scene.Scene, size=config.window_size):
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
        self.scene = scene
        self.pressed_keys = set()
        self.title = f"Manim Community {__version__} - {scene}"
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

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        super().on_mouse_motion(x, y, dx, dy)
        point = self.pixel_coords_to_space_coords(x, y)
        d_point = self.pixel_coords_to_space_coords(dx, dy, relative=True)
        self.scene.on_mouse_motion(point, d_point)

    def on_mouse_drag(
        self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int
    ) -> None:
        super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)
        point = self.pixel_coords_to_space_coords(x, y)
        d_point = self.pixel_coords_to_space_coords(dx, dy, relative=True)
        self.scene.on_mouse_drag(point, d_point, buttons, modifiers)

    def on_mouse_press(self, x: int, y: int, button: int, mods: int) -> None:
        super().on_mouse_press(x, y, button, mods)
        point = self.pixel_coords_to_space_coords(x, y)
        self.scene.on_mouse_press(point, button, mods)

    def on_mouse_release(self, x: int, y: int, button: int, mods: int) -> None:
        super().on_mouse_release(x, y, button, mods)
        point = self.pixel_coords_to_space_coords(x, y)
        self.scene.on_mouse_release(point, button, mods)

    def on_mouse_scroll(self, x: int, y: int, x_offset: float, y_offset: float) -> None:
        super().on_mouse_scroll(x, y, x_offset, y_offset)
        point = self.pixel_coords_to_space_coords(x, y)
        offset = self.pixel_coords_to_space_coords(x_offset, y_offset, relative=True)
        self.scene.on_mouse_scroll(point, offset)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        self.pressed_keys.add(symbol)  # Modifiers?
        super().on_key_press(symbol, modifiers)
        self.scene.on_key_press(symbol, modifiers)

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        self.pressed_keys.difference_update({symbol})  # Modifiers?
        super().on_key_release(symbol, modifiers)
        self.scene.on_key_release(symbol, modifiers)

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        self.scene.on_resize(width, height)

    def on_show(self) -> None:
        super().on_show()
        self.scene.on_show()

    def on_hide(self) -> None:
        super().on_hide()
        self.scene.on_hide()

    def on_close(self) -> None:
        super().on_close()
        self.scene.on_close()

    def is_key_pressed(self, symbol: int) -> bool:
        return symbol in self.pressed_keys
