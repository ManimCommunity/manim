"""Mobject representing highlighted source code listings."""

from __future__ import annotations

__all__ = [
    "Code",
]

from pathlib import Path
from typing import Any, Literal

from bs4 import BeautifulSoup, Tag
from pygments import highlight
from pygments.formatters.html import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer, guess_lexer_for_filename
from pygments.styles import get_all_styles

from manim.constants import *
from manim.mobject.geometry.arc import Dot
from manim.mobject.geometry.shape_matchers import SurroundingRectangle
from manim.mobject.text.text_mobject import Paragraph
from manim.mobject.types.vectorized_mobject import VGroup, VMobject
from manim.typing import StrPath
from manim.utils.color import WHITE, ManimColor


class Code(VMobject):
    """A highlighted source code listing.

    Examples
    --------

    Normal usage::

        listing = Code(
            "helloworldcpp.cpp",
            tab_width=4,
            formatter_style="emacs",
            background="window",
            language="cpp",
            background_config={"stroke_color": WHITE},
            paragraph_config={"font": "Noto Sans Mono"},
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
                self.wait()'''

                rendered_code = Code(
                    code_string=code,
                    language="python",
                    background="window",
                    background_config={"stroke_color": "maroon"},
                )
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
        :meth:`.Code.get_styles_list`.
    tab_width
        The width of a tab character in spaces. Defaults to 4.
    add_line_numbers
        Whether to display line numbers. Defaults to ``True``.
    line_numbers_from
        The first line number to display. Defaults to 1.
    background
        The type of background to use. Can be either ``"rectangle"`` (the
        default) or ``"window"``.
    background_config
        Keyword arguments passed to the background constructor. Default
        settings are stored in the class attribute
        :attr:`.default_background_config` (which can also be modified
        directly).
    paragraph_config
        Keyword arguments passed to the constructor of the
        :class:`.Paragraph` objects holding the code, and the line
        numbers. Default settings are stored in the class attribute
        :attr:`.default_paragraph_config` (which can also be modified
        directly).
    """

    _styles_list_cache: list[str] | None = None
    default_background_config: dict[str, Any] = {
        "buff": 0.3,
        "fill_color": ManimColor("#222"),
        "stroke_color": WHITE,
        "corner_radius": 0.2,
        "stroke_width": 1,
        "fill_opacity": 1,
    }
    default_paragraph_config: dict[str, Any] = {
        "font": "Monospace",
        "font_size": 24,
        "line_spacing": 0.5,
        "disable_ligatures": True,
    }

    def __init__(
        self,
        code_file: StrPath | None = None,
        code_string: str | None = None,
        language: str | None = None,
        formatter_style: str = "vim",
        tab_width: int = 4,
        add_line_numbers: bool = True,
        line_numbers_from: int = 1,
        background: Literal["rectangle", "window"] = "rectangle",
        background_config: dict[str, Any] | None = None,
        paragraph_config: dict[str, Any] | None = None,
    ):
        super().__init__()

        if code_file is not None:
            code_file = Path(code_file)
            code_string = code_file.read_text(encoding="utf-8")
            lexer = guess_lexer_for_filename(code_file.name, code_string)
        elif code_string is not None:
            if language is not None:
                lexer = get_lexer_by_name(language)
            else:
                lexer = guess_lexer(code_string)
        else:
            raise ValueError("Either a code file or a code string must be specified.")

        code_string = code_string.expandtabs(tabsize=tab_width)

        formatter = HtmlFormatter(
            style=formatter_style,
            noclasses=True,
            cssclasses="",
        )
        soup = BeautifulSoup(
            highlight(code_string, lexer, formatter), features="html.parser"
        )
        self._code_html = soup.find("pre")
        assert isinstance(self._code_html, Tag)

        # as we are using Paragraph to render the text, we need to find the character indices
        # of the segments of changed color in the HTML code
        color_ranges = []
        current_line_color_ranges = []
        current_line_char_index = 0
        for child in self._code_html.children:
            if child.name == "span":
                try:
                    child_style = child["style"]
                    if isinstance(child_style, str):
                        color = child_style.removeprefix("color: ")
                    else:
                        color = None
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

        if paragraph_config is None:
            paragraph_config = {}
        base_paragraph_config = self.default_paragraph_config.copy()
        base_paragraph_config.update(paragraph_config)

        self.code_lines = Paragraph(
            *code_lines,
            **base_paragraph_config,
        )
        for line, color_range in zip(self.code_lines, color_ranges):
            for start, end, color in color_range:
                line[start:end].set_color(color)

        if add_line_numbers:
            base_paragraph_config.update({"alignment": "right"})
            self.line_numbers = Paragraph(
                *[
                    str(i)
                    for i in range(
                        line_numbers_from, line_numbers_from + len(self.code_lines)
                    )
                ],
                **base_paragraph_config,
            )
            self.line_numbers.next_to(self.code_lines, direction=LEFT).align_to(
                self.code_lines, UP
            )
            self.add(self.line_numbers)

        self.add(self.code_lines)

        if background_config is None:
            background_config = {}
        background_config_base = self.default_background_config.copy()
        background_config_base.update(background_config)

        if background == "rectangle":
            self.background = SurroundingRectangle(
                self,
                **background_config_base,
            )
        elif background == "window":
            buttons = VGroup(
                Dot(radius=0.1, stroke_width=0, color=button_color)
                for button_color in ["#ff5f56", "#ffbd2e", "#27c93f"]
            ).arrange(RIGHT, buff=0.1)
            buttons.next_to(self, UP, buff=0.1).align_to(self, LEFT).shift(LEFT * 0.1)
            self.background = SurroundingRectangle(
                VGroup(self, buttons),
                **background_config_base,
            )
            buttons.shift(UP * 0.1 + LEFT * 0.1)
            self.background.add(buttons)
        else:
            raise ValueError(f"Unknown background type: {background}")

        self.add_to_back(self.background)

    @classmethod
    def get_styles_list(cls) -> list[str]:
        """Get the list of all available formatter styles."""
        if cls._styles_list_cache is None:
            cls._styles_list_cache = list(get_all_styles())
        return cls._styles_list_cache
