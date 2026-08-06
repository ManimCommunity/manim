"""Mobject representing highlighted source code listings."""

from __future__ import annotations

__all__ = [
    "Code",
]

import re
from pathlib import Path
from typing import Any, Literal

from bs4 import BeautifulSoup, Tag
from pygments import highlight
from pygments.formatters.html import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer, guess_lexer_for_filename
from pygments.styles import get_all_styles, get_style_by_name
from pygments.token import Text as TextToken

from manim.constants import *
from manim.mobject.geometry.arc import Dot
from manim.mobject.geometry.shape_matchers import SurroundingRectangle
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.types.vectorized_mobject import VGroup, VMobject
from manim.typing import StrPath
from manim.utils.color import BLACK, WHITE, ManimColor


class Code(VMobject, metaclass=ConvertToOpenGL):
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
        directly). If ``fill_color`` is not specified, it is taken from
        the selected ``formatter_style``.
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
        "fill_color": None,
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
    code: VMobject

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
            if language is not None:
                lexer = get_lexer_by_name(language)
            else:
                lexer = guess_lexer_for_filename(code_file.name, code_string)
        elif code_string is not None:
            if language is not None:
                lexer = get_lexer_by_name(language)
            else:
                lexer = guess_lexer(code_string)
        else:
            raise ValueError("Either a code file or a code string must be specified.")

        code_string = code_string.expandtabs(tabsize=tab_width)

        selected_style = get_style_by_name(formatter_style)
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

        code_lines = self._code_html.get_text().removesuffix("\n").split("\n")

        base_paragraph_config = self.default_paragraph_config.copy()
        base_paragraph_config.update(paragraph_config or {})
        paragraph_color_is_configured = "color" in base_paragraph_config
        default_text_color = selected_style.style_for_token(TextToken).get("color")
        if default_text_color is not None:
            default_text_color = ManimColor(f"#{default_text_color}")
        base_paragraph_config.setdefault("color", default_text_color or BLACK)

        from manim.mobject.text.text_mobject import Paragraph

        self.code_lines = Paragraph(
            *code_lines,
            **base_paragraph_config,
        )

        i_line, i_char = 0, 0
        for child in self._code_html.children:
            if (
                isinstance(child, Tag)
                and child.name == "span"
                and child.has_attr("style")
            ):
                match_ = re.match(
                    r"color: (#[A-Fa-f0-9]{6}|#[A-Fa-f0-9]{3})", child["style"]
                )
                if match_ is not None:
                    self.code_lines[i_line][
                        i_char : i_char + len(child.text)
                    ].set_color(match_.group(1))

            for char in child.text:
                if char == "\n":
                    i_line += 1
                    i_char = 0
                else:
                    i_char += 1

        if add_line_numbers:
            line_number_config = base_paragraph_config.copy()
            line_number_config["alignment"] = "right"
            if not paragraph_color_is_configured:
                line_number_color = selected_style.line_number_color
                if line_number_color != "inherit":
                    line_number_config["color"] = line_number_color
            self.line_numbers = Paragraph(
                *[
                    str(i)
                    for i in range(
                        line_numbers_from, line_numbers_from + len(self.code_lines)
                    )
                ],
                **line_number_config,
            )
            self.line_numbers.next_to(self.code_lines, direction=LEFT).align_to(
                self.code_lines, UP
            )
            self.add(self.line_numbers)

        for line in self.code_lines:
            line.submobjects = [c for c in line if not isinstance(c, Dot)]
        self.add(self.code_lines)

        background_config_base = self.default_background_config.copy()
        background_config_base.update(background_config or {})
        if background_config_base["fill_color"] is None:
            background_config_base["fill_color"] = selected_style.background_color

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
