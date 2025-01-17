from manim import *
from manim.mobject.text.code_transform import CodeTransform
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "code_transform"


@frames_comparison
def test_code_transform(scene):
    before_code = """print("Hello, World!")"""
    after_code = """print("Hello, Manim!")"""

    before = Code(
        code=before_code,
        language="Python",
    )

    after = Code(
        code=after_code,
        language="Python",
    )

    scene.add(before)
    scene.play(CodeTransform(before, after))
