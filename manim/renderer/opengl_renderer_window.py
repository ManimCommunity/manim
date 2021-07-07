import moderngl_window as mglw
from moderngl_window.context.pyglet.window import Window as PygletWindow
from moderngl_window.timers.clock import Timer
from screeninfo import get_monitors

from .. import __version__, config


class Window(PygletWindow):
    fullscreen = False
    resizable = True
    gl_version = (3, 3)
    vsync = True
    cursor = True

    def __init__(self, renderer, size=None, **kwargs):
        if size is None:
            # Default to making window half the screen size
            # but make it full screen if --fullscreen is passed in
            monitors = get_monitors()
            mon_index = config.window_monitor
            monitor = monitors[min(mon_index, len(monitors) - 1)]
            window_width = monitor.width

            if not config.fullscreen:
                window_width //= 2

            window_height = int(
                window_width * config.frame_height // config.frame_width
            )
            size = (window_width, window_height)

        super().__init__(size=size)

        self.title = f"Manim Community {__version__}"
        self.size = size
        self.renderer = renderer

        mglw.activate_context(window=self)
        self.timer = Timer()
        self.config = mglw.WindowConfig(ctx=self.ctx, wnd=self, timer=self.timer)
        self.timer.start()

        self.swap_buffers()

        initial_position = self.find_initial_position(size, monitor)
        self.position = initial_position

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

    def find_initial_position(self, size, monitor):
        custom_position = config.window_position
        window_width, window_height = size
        # Position might be specified with a string of the form
        # x,y for integers x and y
        if len(custom_position) == 1:
            raise ValueError(
                "window_position must specify both Y and X positions (Y/X -> UR). Also accepts LEFT/RIGHT/ORIGIN/UP/DOWN."
            )
        # in the form Y/X (UR)
        if custom_position == "LEFT" or custom_position == "RIGHT":
            custom_position = "O" + custom_position[0]
        elif custom_position == "UP" or custom_position == "DOWN":
            custom_position = custom_position[0] + "O"
        elif custom_position == "ORIGIN":
            custom_position = "O" * 2
        elif "," in custom_position:
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
