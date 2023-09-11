import time

import pyglet
from PIL import Image
from pyglet import shapes
from pyglet.gl import Config
from pyglet.window import Window

import manim.utils.color.manim_colors as col
from manim._config import tempconfig
from manim.camera.camera import OpenGLCamera, OpenGLCameraFrame
from manim.constants import OUT, RIGHT
from manim.mobject.geometry.arc import Circle
from manim.mobject.geometry.polygram import Square
from manim.mobject.logo import ManimBanner
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
from manim.renderer.opengl_renderer import OpenGLRenderer

if __name__ == "__main__":
    with tempconfig({"renderer": "opengl"}):
        renderer = OpenGLRenderer(1920, 1080)
        # vm = OpenGLVMobject([col.RED, col.GREEN])
        vm = Circle(
            radius=1, stroke_color=col.YELLOW, fill_opacity=1, fill_color=col.RED
        ).shift(RIGHT)
        vm2 = Square(stroke_color=col.GREEN, fill_opacity=0, stroke_opacity=1)
        # vm3 = ManimBanner()
        # vm.set_points_as_corners([[-1920/2, 0, 0], [1920/2, 0, 0], [0, 1080/2, 0]])
        # print(vm.color)
        # print(vm.fill_color)
        # print(vm.stroke_color)

        camera = OpenGLCameraFrame()
        renderer.init_camera(camera)

        renderer.render(camera, [vm, vm2])
        image = renderer.get_pixels()
        print(image.shape)
        Image.fromarray(image, "RGBA").show()
        exit(0)
        win = Window(
            width=1920,
            height=1080,
            vsync=True,
            config=Config(double_buffer=True, samples=4),
        )
        renderer.use_window_fbo()

        @win.event
        def on_close():
            win.close()
            pass

        @win.event
        def on_mouse_motion(x, y, dx, dy):
            vm.move_to((14.2222 * (x / 1920 - 0.5), 8 * (y / 1080 - 0.5), 0))
            # vm.set_color(col.RED.interpolate(col.GREEN,x/1920))
            # print(x,y)

        @win.event
        def on_draw():
            image = renderer.render(camera, [vm, vm2])
            pass

        @win.event
        def on_resize(width, height):
            pass

        while True:
            pyglet.clock.tick()
            pyglet.app.platform_event_loop.step()
            win.switch_to()
            win.dispatch_event("on_draw")
            win.dispatch_events()
            win.flip()
