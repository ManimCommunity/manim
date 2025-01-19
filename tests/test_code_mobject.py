from manim.mobject.text.code_mobject import Code
from manim.utils.color.core import ManimColor


def test_code_initialization_from_string():
    code_string = """from manim import Scene, Square

class FadeInSquare(Scene):
    def construct(self):
        s = Square()
        self.play(FadeIn(s))
        self.play(s.animate.scale(2))
        self.wait()"""
    rendered_code = Code(
        code_string=code_string,
        language="python",
    )
    num_lines = len(code_string.split("\n"))
    assert len(rendered_code.code_lines) == num_lines
    assert len(rendered_code.line_numbers) == num_lines


def test_code_initialization_from_file():
    rendered_code = Code(
        code_file="tests/test_code_mobject.py",
        language="python",
        background="window",
        background_fill_color="#101010",
    )
    assert len(rendered_code.code_lines) == len(rendered_code.line_numbers)
    assert rendered_code.background.fill_color == ManimColor("#101010")
