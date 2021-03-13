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

    def __init__(self, size=None, **kwargs):
        if size is None:
            size = (config["pixel_width"], config["pixel_height"])
        super().__init__(size=size)

        self.pressed_keys = set()

        self.title = f"ManimCommunity {__version__}"
        self.size = size

        mglw.activate_context(window=self)
        self.timer = Timer()
        self.config = mglw.WindowConfig(ctx=self.ctx, wnd=self, timer=self.timer)
        self.timer.start()

        self.swap_buffers()
