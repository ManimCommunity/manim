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


@pytest.mark.parametrize(
    ("code_string", "num_lines"),
    [("pass", 1), ("   ", 1), (" \n ", 2)],
)
def test_code_initialization_without_line_numbers(code_string, num_lines):
    rendered_code = Code(
        code_string=code_string,
        language="python",
        add_line_numbers=False,
    )

    assert len(rendered_code.code_lines) == num_lines
    assert not hasattr(rendered_code, "line_numbers")


def test_code_initialization_from_file():
    rendered_code = Code(
        code_file="example_scenes/basic.py",
        language="python",
        background="window",
        background_config={"fill_color": "#101010"},
        paragraph_config={"font_size": 14},
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


def test_code_ignores_paragraph_color():
    rendered_code = Code(
        code_string="plain text",
        language="text",
        formatter_style="vim",
        paragraph_config={"color": "#ff0000"},
    )

    assert rendered_code.code_lines[0][0].get_color() == ManimColor("#cccccc")
    assert rendered_code.line_numbers[0][0].get_color() == ManimColor("#cccccc")


def test_code_ignores_default_paragraph_color(monkeypatch):
    monkeypatch.setitem(Code.default_paragraph_config, "color", "#ff0000")

    rendered_code = Code(
        code_string="plain text",
        language="text",
        formatter_style="vim",
    )

    assert rendered_code.code_lines[0][0].get_color() == ManimColor("#cccccc")
    assert rendered_code.line_numbers[0][0].get_color() == ManimColor("#cccccc")


def test_code_accepts_custom_pygments_style():
    class CustomStyle(Code.get_pygments_style("xcode")):
        background_color = "#123456"
        line_number_color = "#abcdef"

    rendered_code = Code(
        code_string="plain text",
        language="text",
        formatter_style=CustomStyle,
    )

    assert rendered_code.background.fill_color == ManimColor("#123456")
    assert rendered_code.code_lines[0][0].get_color() == ManimColor("#000000")
    assert rendered_code.line_numbers[0][0].get_color() == ManimColor("#abcdef")


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


@pytest.mark.parametrize("line_numbers_from", [1, 10, -2])
def test_code_baseline_alignment(line_numbers_from):
    baseline_offsets = []
    for code_string in ("pass", "pass b"):
        rendered_code = Code(
            code_string=code_string,
            language="python",
            line_numbers_from=line_numbers_from,
        )
        baseline_offsets.append(
            rendered_code.code_lines[0].get_bottom()[1]
            - rendered_code.line_numbers[0].get_bottom()[1]
        )

    np.testing.assert_allclose(baseline_offsets[0], baseline_offsets[1], atol=1e-6)


@pytest.mark.parametrize("background", ["rectangle", "window"])
@pytest.mark.parametrize("add_line_numbers", [True, False])
def test_code_height_is_independent_of_boundary_glyphs(
    background,
    add_line_numbers,
):
    code_strings = (
        "AAA\nppp",
        "ppp\nAAA",
        "AAA\nAAA",
        "ppp\nppp",
        "---\n---",
    )
    background_heights = [
        Code(
            code_string=code_string,
            language="python",
            background=background,
            add_line_numbers=add_line_numbers,
        ).background.height
        for code_string in code_strings
    ]

    np.testing.assert_allclose(background_heights, background_heights[0], atol=1e-6)


def test_code_syntax_highlighting_colors():
    code_string = "pass\n# comment"
    rendered_code = Code(
        code_string=code_string,
        language="python",
        formatter_style="vim",
    )

    assert {char.color for char in rendered_code.code_lines[0]} == {
        ManimColor("#CDCD00")
    }
    assert {char.color for char in rendered_code.code_lines[1]} == {
        ManimColor("#000080")
    }
    assert [len(line) for line in rendered_code.code_lines] == [
        sum(not char.isspace() for char in line) for line in code_string.splitlines()
    ]
    assert len(rendered_code.code_lines) == len(rendered_code.code_lines.chars)


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
