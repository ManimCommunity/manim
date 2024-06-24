from manim import manim_colors as col
from manim.mobject.types.vectorized_mobject import OpenGLVMobject

VMobject = OpenGLVMobject


def test_vmobject_init():
    vm = VMobject(col.RED)
    assert vm.fill_color == [col.RED]
    assert vm.stroke_color == [col.RED]
    vm = VMobject(col.GREEN, stroke_color=col.YELLOW)
    assert vm.fill_color == [col.GREEN]
    assert vm.stroke_color == [col.YELLOW]
    vm = VMobject()
    assert vm.fill_color == [col.WHITE]
    assert vm.stroke_color == [col.WHITE]
