from .. import config, __version__
import moderngl_window as mglw
from moderngl_window.context.pyglet.window import Window as PygletWindow
from moderngl_window.timers.clock import Timer


class Window(PygletWindow):
    fullscreen = False
    resizable = True
    gl_version = (3, 3)
    vsync = True
    cursor = True

    def __init__(self, renderer, size=None, **kwargs):
        if size is None:
            size = (config["pixel_width"], config["pixel_height"])
        super().__init__(size=size)

        self.pressed_keys = set()

        self.title = f"Manim Community {__version__}"
        self.size = size
        self.renderer = renderer

        mglw.activate_context(window=self)
        self.timer = Timer()
        self.config = mglw.WindowConfig(ctx=self.ctx, wnd=self, timer=self.timer)
        self.timer.start()

        self.swap_buffers()

    # Delegate event handling to scene
    def on_mouse_motion(self, x, y, dx, dy):
        super().on_mouse_motion(x, y, dx, dy)
        point = self.renderer.pixel_coords_to_space_coords(x, y)
        d_point = self.renderer.pixel_coords_to_space_coords(dx, dy, relative=True)
        self.renderer.scene.on_mouse_motion(point, d_point)
