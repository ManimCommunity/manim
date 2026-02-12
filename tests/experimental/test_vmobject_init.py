from manim import manim_colors as col
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject as VMobject


def test_vmobject_init():
    vm = VMobject()
    assert vm.fill_color == [col.WHITE]
    assert vm.stroke_color == [col.WHITE]
    vm = VMobject(color=col.RED)
    assert vm.fill_color == [col.RED]
    assert vm.stroke_color == [col.RED]
    vm = VMobject(fill_color=col.GREEN, stroke_color=col.YELLOW)
    assert vm.fill_color == [col.GREEN]
    assert vm.stroke_color == [col.YELLOW]
