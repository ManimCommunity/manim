import numpy as np
import pytest

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
        background_config={"fill_color": "#101010"},
    )
    assert len(rendered_code.code_lines) == len(rendered_code.line_numbers)
    assert rendered_code.background.fill_color == ManimColor("#101010")


@pytest.mark.parametrize(
    ("formatter_style", "background_color", "text_color"),
    [
        ("xcode", "#ffffff", "#000000"),
        ("vim", "#000000", "#cccccc"),
    ],
)
def test_code_uses_formatter_style_colors(
    formatter_style, background_color, text_color
):
    rendered_code = Code(
        code_string="plain text",
        language="text",
        formatter_style=formatter_style,
        add_line_numbers=False,
    )

    assert rendered_code.background.fill_color == ManimColor(background_color)
    assert rendered_code.code_lines[0][0].get_color() == ManimColor(text_color)


def test_code_paragraph_color_overrides_formatter_style_color():
    rendered_code = Code(
        code_string="plain text",
        language="text",
        formatter_style="vim",
        paragraph_config={"color": "#ff0000"},
    )

    assert rendered_code.code_lines[0][0].get_color() == ManimColor("#ff0000")
    assert rendered_code.line_numbers[0][0].get_color() == ManimColor("#ff0000")


def test_code_default_paragraph_color_overrides_formatter_style_color(monkeypatch):
    monkeypatch.setitem(Code.default_paragraph_config, "color", "#ff0000")

    rendered_code = Code(
        code_string="plain text",
        language="text",
        formatter_style="vim",
    )

    assert rendered_code.code_lines[0][0].get_color() == ManimColor("#ff0000")
    assert rendered_code.line_numbers[0][0].get_color() == ManimColor("#ff0000")


@pytest.mark.parametrize(
    ("formatter_style", "line_number_color"),
    [
        ("solarized-light", ManimColor("#93a1a1")),
        ("vim", ManimColor("#cccccc")),
        ("xcode", ManimColor("#000000")),
    ],
)
def test_code_line_numbers_use_formatter_style_color(
    formatter_style, line_number_color
):
    rendered_code = Code(
        code_string="plain text",
        language="text",
        formatter_style=formatter_style,
    )

    assert rendered_code.line_numbers[0][0].get_color() == line_number_color


def test_line_heights_initial_whitespace():
    rendered_code = Code(
        code_string="""print('Hello, World!')
for _ in range(42):
    print('Hello, World!')
""",
        language="python",
    )
    np.testing.assert_almost_equal(
        rendered_code.code_lines[0].height,
        rendered_code.code_lines[2].height,
    )


def test_code_initialization_style_correct_color():
    for style in Code.get_styles_list():
        try:
            Code(
                code_string="""# This is a comment.
var = 3
print(var)
""",
                formatter_style=style,
            )
        except ValueError as e:
            pytest.fail(f"Code initialization failed for style {style} with error: {e}")
