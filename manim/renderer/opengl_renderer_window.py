import moderngl_window as mglw
from moderngl_window.context.pyglet.window import Window as PygletWindow
from moderngl_window.timers.clock import Timer

from .. import __version__, config


class Window(PygletWindow):
    fullscreen = False
    resizable = True
    gl_version = (3, 3)
    vsync = True
    cursor = True

    def __init__(self, renderer, size=None, samples=0, **kwargs):
        if size is None:
            size = (config["pixel_width"], config["pixel_height"])
        super().__init__(size=size, samples=samples)

        self.title = f"Manim Community {__version__}"
        self.size = size
        self.renderer = renderer

        mglw.activate_context(window=self)
        self.timer = Timer()
        self.config = mglw.WindowConfig(ctx=self.ctx, wnd=self, timer=self.timer)
        self.timer.start()

        self.swap_buffers()

    # Delegate event handling to scene.
    def on_mouse_motion(self, x, y, dx, dy):
        super().on_mouse_motion(x, y, dx, dy)
        point = self.renderer.pixel_coords_to_space_coords(x, y)
        d_point = self.renderer.pixel_coords_to_space_coords(dx, dy, relative=True)
        self.renderer.scene.on_mouse_motion(point, d_point)

    def on_mouse_scroll(self, x, y, x_offset: float, y_offset: float):
        super().on_mouse_scroll(x, y, x_offset, y_offset)
        point = self.renderer.pixel_coords_to_space_coords(x, y)
        offset = self.renderer.pixel_coords_to_space_coords(
            x_offset, y_offset, relative=True
        )
        self.renderer.scene.on_mouse_scroll(point, offset)

    def on_key_press(self, symbol, modifiers):
        self.renderer.pressed_keys.add(symbol)
        super().on_key_press(symbol, modifiers)
        self.renderer.scene.on_key_press(symbol, modifiers)

    def on_key_release(self, symbol, modifiers):
        if symbol in self.renderer.pressed_keys:
            self.renderer.pressed_keys.remove(symbol)
        super().on_key_release(symbol, modifiers)
        self.renderer.scene.on_key_release(symbol, modifiers)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)
        point = self.renderer.pixel_coords_to_space_coords(x, y)
        d_point = self.renderer.pixel_coords_to_space_coords(dx, dy, relative=True)
        self.renderer.scene.on_mouse_drag(point, d_point, buttons, modifiers)
