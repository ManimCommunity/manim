"""Mobject representing highlighted source code listings."""

from __future__ import annotations

__all__ = [
    "Code",
]

import os

from pathlib import Path
from bs4 import BeautifulSoup

from pygments import highlight
from pygments.formatters.html import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer_for_filename, guess_lexer

from manim.constants import *
from manim.mobject.geometry.arc import Dot
from manim.mobject.geometry.shape_matchers import SurroundingRectangle
from manim.mobject.text.text_mobject import Paragraph
from manim.mobject.types.vectorized_mobject import VGroup, VMobject


class Code(VMobject):
    """A highlighted source code listing.

    Examples
    --------

    Normal usage::

        listing = Code(
            "helloworldcpp.cpp",
            tab_width=4,
            background_stroke_width=1,
            background_stroke_color=WHITE,
            insert_line_no=True,
            style="emacs",
            background="window",
            language="cpp",
        )

    We can also render code passed as a string. As the automatic language
    detection can be a bit flaky, it is recommended to specify the language
    explicitly:

    .. manim:: CodeFromString
        :save_last_frame:

        class CodeFromString(Scene):
            def construct(self):
                code = '''from manim import Scene, Square

        class FadeInSquare(Scene):
            def construct(self):
                s = Square()
                self.play(FadeIn(s))
                self.play(s.animate.scale(2))
                self.wait()
        '''
                rendered_code = Code(code=code, tab_width=4, background="window",
                                    language="Python", font="Monospace")
                self.add(rendered_code)

    Parameters
    ----------
    code_file
        The path to the code file to display.
    code_string
        Alternatively, the code string to display.
    language
        The programming language of the code. If not specified, it will be
        guessed from the file extension or the code itself.
    formatter_style
        The style to use for the code highlighting. Defaults to ``"vim"``.
        A list of all available styles can be obtained by calling
        :func:`pygments.styles.get_all_styles`.
    line_numbers
        Whether to display line numbers. Defaults to ``True``.
    line_numbers_from
        The first line number to display. Defaults to 1.
    background
        The type of background to use. Can be either ``"rectangle"`` (the
        default) or ``"window"``.
    background_margin
        The margin between the code and the background in Manim units.
        Defaults to 0.3.
    background_stroke_color
        The color of the border of the background. Defaults to ``"#fff"``.
    background_fill_color
        The color of the background. Defaults to ``"#222"``.
    background_corner_radius
        The corner radius of the background. Defaults to 0.2.
    font
        The font to use for the code. Defaults to ``"Monospace"``.
    font_size
        The font size to use for the code. Defaults to 24.
    line_spacing
        The amount of space between lines in relation to the font size.
        Defaults to 0.5.
    **paragraph_kwargs
        Additional keyword arguments passed to :class:`.Paragraph`.
    """

    def __init__(
        self,
        code_file: os.PathLike | str | None = None,
        code_string: str | None = None,
        language: str | None = None,
        formatter_style: str = "vim",
        line_numbers: bool = True,
        line_numbers_from: int = 1,
        background: str = "rectangle",
        background_margin: float = 0.3,
        background_stroke_color: str = "#fff",
        background_fill_color: str = "#222",
        background_corner_radius: float = 0.2,
        font: str = "Monospace",
        font_size: float = 24.0,
        line_spacing: float = 0.5,
        **paragraph_kwargs,
    ):
        super().__init__()

        # collect all arguments for Paragraph initialization in one dict
        paragraph_kwargs.update(
            {
                "font": font,
                "font_size": font_size,
                "line_spacing": line_spacing,
                "disable_ligatures": True,
            }
        )

        if code_file is None and code_string is None:
            raise ValueError("Either a code file or a code string must be specified.")

        if code_file is not None:
            code_file = Path(code_file)
            code_string = code_file.read_text(encoding="utf-8")
            lexer = guess_lexer_for_filename(code_file.name, code_string)
        else:
            if language is not None:
                lexer = get_lexer_by_name(language)
            else:
                lexer = guess_lexer(code_string)

        formatter = HtmlFormatter(
            style=formatter_style,
            noclasses=True,
            cssclasses="",
        )
        soup = BeautifulSoup(
            highlight(code_string, lexer, formatter), features="html.parser"
        )
        self._code_html = soup.find("pre")

        # as we are using Paragraph to render the text, we need to find the character indices
        # of the segments of changed color in the HTML code
        color_ranges = []
        current_line_color_ranges = []
        current_line_char_index = 0
        for child in self._code_html.children:
            if child.name == "span":
                try:
                    color = child["style"].removeprefix("color: ")
                except KeyError:
                    color = None
                current_line_color_ranges.append(
                    (
                        current_line_char_index,
                        current_line_char_index + len(child.text),
                        color,
                    )
                )
                current_line_char_index += len(child.text)
            else:
                for char in child.text:
                    if char == "\n":
                        color_ranges.append(current_line_color_ranges)
                        current_line_color_ranges = []
                        current_line_char_index = 0
                    else:
                        current_line_char_index += 1

        color_ranges.append(current_line_color_ranges)

        code_lines = self._code_html.get_text().removesuffix("\n").split("\n")
        self.code_lines = Paragraph(
            *code_lines,
            **paragraph_kwargs,
        )
        for line, color_range in zip(self.code_lines, color_ranges):
            for start, end, color in color_range:
                line[start:end].set_color(color)

        if line_numbers:
            paragraph_kwargs.update(
                {
                    "alignment": "right",
                }
            )
            self.line_numbers = Paragraph(
                *[
                    str(i)
                    for i in range(
                        line_numbers_from, line_numbers_from + len(self.code_lines)
                    )
                ],
                **paragraph_kwargs,
            )
            self.line_numbers.next_to(self.code_lines, direction=LEFT).align_to(
                self.code_lines, UP
            )
            self.add(self.line_numbers)

        self.add(self.code_lines)

        if background == "rectangle":
            self.background = SurroundingRectangle(
                self,
                buff=background_margin,
                fill_color=background_fill_color,
                stroke_color=background_stroke_color,
                corner_radius=background_corner_radius,
                stroke_width=1,
                fill_opacity=1,
            )
        elif background == "window":
            buttons = VGroup(
                Dot(radius=0.1, stroke_width=0, color=button_color)
                for button_color in ["#ff5f56", "#ffbd2e", "#27c93f"]
            ).arrange(RIGHT, buff=0.1)
            buttons.next_to(self, UP, buff=0.1).align_to(self, LEFT).shift(LEFT * 0.1)
            self.background = SurroundingRectangle(
                VGroup(self, buttons),
                buff=background_margin,
                fill_color=background_fill_color,
                stroke_color=background_stroke_color,
                corner_radius=background_corner_radius,
                stroke_width=1,
                fill_opacity=1,
            )
            buttons.shift(UP * 0.1 + LEFT * 0.1)
            self.background.add(buttons)
        else:
            raise ValueError(f"Unknown background type: {background}")

        self.add_to_back(self.background)
