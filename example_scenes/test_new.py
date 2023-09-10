import manim.utils.color.manim_colors as col
from manim.camera.camera import OpenGLCamera, OpenGLCameraFrame
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
from manim.renderer.opengl_renderer import OpenGLRenderer

renderer = OpenGLRenderer(1920, 1080)
vm = OpenGLVMobject([col.RED, col.GREEN])
vm.set_points_as_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
# print(vm.color)
# print(vm.fill_color)
# print(vm.stroke_color)

camera = OpenGLCameraFrame((1920, 1090))
renderer.render_vmobject(vm, camera)
