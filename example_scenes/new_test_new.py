import time

import numpy as np

# import pyglet
from pyglet.gl import Config
from pyglet.window import Window

import manim.utils.color.manim_colors as col
from manim._config import tempconfig
from manim.animation.creation import DrawBorderThenFill
from manim.camera.camera import Camera
from manim.constants import LEFT, OUT, RIGHT, UP
from manim.mobject.geometry.arc import Circle
from manim.mobject.geometry.polygram import Square
from manim.mobject.logo import ManimBanner
from manim.mobject.text.numbers import DecimalNumber
from manim.renderer.opengl_renderer import OpenGLRenderer

if __name__ == "__main__":
    with tempconfig({"renderer": "opengl"}):
        win = Window(
            width=1920,
            height=1080,
            fullscreen=True,
            vsync=True,
            config=Config(double_buffer=True, samples=0),
        )
        renderer = OpenGLRenderer(1920, 1080, background_color=col.GRAY)
        # vm = OpenGLVMobject([col.RED, col.GREEN])
        vm = (
            Circle(
                radius=1,
                stroke_color=col.YELLOW,
            )
            .shift(3 * RIGHT + OUT)
            .set_opacity(0.6)
        )
        vm2 = Square(stroke_color=col.GREEN, fill_opacity=0, stroke_opacity=1).move_to(
            (0, 0, -0.5)
        )
        vm3 = ManimBanner().set_opacity(0.6)
        vm4 = (
            Circle(0.5, col.GREEN)
            .set_opacity(0.6)
            .shift(OUT)
            .set_fill(col.BLUE, opacity=0.2)
        )
        # vm.set_points_as_corners([[-1920/2, 0, 0], [1920/2, 0, 0], [0, 1080/2, 0]])
        # print(vm.color)
        # print(vm.fill_color)
        # print(vm.stroke_color)

        clock_mobject = DecimalNumber(0.0).shift(4 * LEFT + 2.5 * UP)
        clock_mobject.fix_in_frame()

        camera = Camera()
        camera.save_state()
        # renderer.init_camera(camera)

        # renderer.render(camera, [vm, vm2])
        # image = renderer.get_pixels()
        # print(image.shape)
        # Image.fromarray(image, "RGBA").show()
        # exit(0)
        renderer.use_window()

        # clock = pyglet.clock.get_default()

        def update_circle(dt):
            vm.move_to((np.sin(dt) * 4, np.cos(dt) * 4, -1))

        def p2m(x, y, z):
            from manim._config import config

            return (
                config.frame_width * (x / config.pixel_width - 0.5),
                config.frame_height * (y / config.pixel_height - 0.5),
                z,
            )

        @win.event
        def on_close():
            win.close()

        @win.event
        def on_mouse_motion(x, y, dx, dy):
            # vm.move_to((14.2222 * (x / 1920 - 0.5), 8 * (y / 1080 - 0.5), 0))
            # camera.move_to(p2m(x,y,camera.get_center()[2]))
            from scipy.spatial.transform import Rotation

            camera.set_orientation(
                Rotation.from_rotvec(
                    (-UP * (x / 1920 - 0.5) + RIGHT * (y / 1080 - 0.5)) * 2 * 3.1415
                )
            )
            # vm.set_color(col.RED.interpolate(col.GREEN,x/1920))
            # print(x,y)

        @win.event
        def on_draw():
            # dt = clock.update_time()
            renderer.render(camera, [vm2, vm3, vm4, clock_mobject, vm])
            # update_circle(counter)

        @win.event
        def on_resize(width, height):
            super(Window, win).on_resize(width, height)

        # pyglet.app.run()
        has_started = False
        is_finished = False

        run_time = 5
        new_vm = Square(fill_color=col.GREEN, stroke_color=col.BLUE).shift(
            2.5 * RIGHT - UP + 2 * OUT
        )
        animation = DrawBorderThenFill(vm3, run_time=run_time)

        real_time = 0
        virtual_time = 0
        start_timestamp = time.time()
        dt = 1 / 30

        while True:
            # pyglet.app.platform_event_loop.step()
            win.switch_to()
            if not has_started:
                animation.begin()
                has_started = True

            real_time = time.time() - start_timestamp
            while virtual_time < real_time:
                virtual_time += dt
                if not is_finished:
                    if virtual_time >= run_time:
                        animation.finish()
                        buffer = str(animation.buffer)
                        print(f"buffer = {buffer}")
                        has_finished = True
                    else:
                        animation.update_mobjects(dt)
                        animation.interpolate(virtual_time / run_time)
            # update_circle(virtual_time)
            clock_mobject.set_value(virtual_time)
            win.dispatch_event("on_draw")
            win.dispatch_events()
            win.flip()
