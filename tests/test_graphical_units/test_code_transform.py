from manim import *
from manim.mobject.text.code_transform import CodeTransform
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "code_transform"


@frames_comparison
def test_code_transform(scene):
    before_code = """from manim import *

class Animation(Scene):
    def construct(self):

        square = Square(side_length=2.0, color=RED)
        square.shift(LEFT * 2)

        self.play(Create(square))
        self.wait()
"""

    after_code = """from manim import *

class Animation(Scene):
    def construct(self):

        circle = Circle(radius=1.0, color=BLUE)
        circle.shift(RIGHT * 2)

        square = Square(side_length=2.0, color=RED)
        square.shift(LEFT * 2)

        self.play(Create(circle))
        self.wait()
"""

    before = Code(
        code=before_code,
        language="Python",
    ).scale(0.8)

    after = Code(
        code=after_code,
        language="Python",
        line_spacing=1,
    ).scale(0.8)

    scene.add(before)
    scene.play(CodeTransform(before, after))
