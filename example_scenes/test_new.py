import time

from PIL import Image
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
        win = Window(width=1920, height=1080)
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
        renderer.set_camera(camera)

        image = renderer.render(camera, [vm, vm2])
        # print(image.shape)
        # Image.fromarray(image,"RGBA").show()
        for _ in range(4):
            image = renderer.render(camera, [vm, vm2])
            win.dispatch_events()
            win.flip()
            time.sleep(1)
