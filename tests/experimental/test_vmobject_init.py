from manim import manim_colors as col
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject


def test_vmobject_init():
    vm = OpenGLVMobject()
    assert vm.fill_color == [col.WHITE]
    assert vm.stroke_color == [col.WHITE]
    vm = OpenGLVMobject(color=col.RED)
    assert vm.fill_color == [col.RED]
    assert vm.stroke_color == [col.RED]
    vm = OpenGLVMobject(fill_color=col.GREEN, stroke_color=col.YELLOW)
    assert vm.fill_color == [col.GREEN]
    assert vm.stroke_color == [col.YELLOW]
