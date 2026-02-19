from manim import manim_colors as col
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject as VMobject


def test_vmobject_init():
    vm = VMobject()
    assert vm.get_fill_color() == col.WHITE
    assert vm.get_stroke_color() == col.WHITE
    vm = VMobject(color=col.RED)
    assert vm.get_fill_color() == col.RED
    assert vm.get_stroke_color() == col.RED
    vm = VMobject(fill_color=col.GREEN, stroke_color=col.YELLOW)
    assert vm.get_fill_color() == col.GREEN
    assert vm.get_stroke_color() == col.YELLOW
