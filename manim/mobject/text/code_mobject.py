"""Mobject representing highlighted source code listings."""

from __future__ import annotations

__all__ = [
    "Code",
]

from pathlib import Path
from typing import Any, Callable, Literal

from bs4 import BeautifulSoup, Tag
from pygments import highlight
from pygments.formatters.html import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer, guess_lexer_for_filename
from pygments.styles import get_all_styles

from manim.animation.composition import AnimationGroup, LaggedStart
from manim.animation.fading import FadeIn, FadeOut
from manim.animation.transform import Transform
from manim.constants import *
from manim.mobject.geometry.arc import Dot
from manim.mobject.geometry.shape_matchers import SurroundingRectangle
from manim.mobject.mobject import override_animate
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.text.text_mobject import Paragraph
from manim.mobject.types.vectorized_mobject import VGroup, VMobject
from manim.typing import StrPath
from manim.utils.color import WHITE, ManimColor
from manim.utils.rate_functions import linear


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
        directly).
    paragraph_config
        Keyword arguments passed to the constructor of the
        :class:`.Paragraph` objects holding the code, and the line
        numbers. Default settings are stored in the class attribute
        :attr:`.default_paragraph_config` (which can also be modified
        directly).
    """

    line_numbers: Paragraph | None = None
    background: VMobject | None = None
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
        self._current_code_string = code_string

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

    def _set_current_code_string(self, new_str: str) -> None:
        self._current_code_string = new_str

    def update_code(
        self,
        code_file: StrPath | None = None,
        code_string: str | None = None,
        language: str | None = None,
    ) -> Code:
        self._target_code_file = code_file
        self._target_code_string = code_string
        self._target_code_language = language
        return self

    @override_animate(update_code)
    def _animate_update_code(
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
        run_time: float | None = None,
        rate_func: Callable[..., float] = linear,
        lag_ratio: float = 0.0,
        **kwargs: Any,
    ) -> AnimationGroup:
        old_code_string = self._current_code_string
        old_lines = list(self.code_lines) if hasattr(self, "code_lines") else []
        old_background = getattr(self, "background", None)
        old_line_numbers = getattr(self, "line_numbers", None)

        if code_file is not None:
            p = Path(code_file)
            new_code_str = p.read_text(encoding="utf-8")
        else:
            new_code_str = "" if code_string is None else code_string

        if not new_code_str:
            raise ValueError("No new code_string or code_file found for update_code.")

        if language is None:
            language = self._target_code_language

        tmp_new_code = type(self)(
            code_file=None,
            code_string=new_code_str,
            language=language,
            formatter_style=formatter_style,
            tab_width=tab_width,
            add_line_numbers=add_line_numbers,
            line_numbers_from=line_numbers_from,
            background=background,
            background_config=background_config,
            paragraph_config=paragraph_config,
        )
        new_lines = list(tmp_new_code.code_lines)
        new_background = tmp_new_code.background
        new_line_numbers = getattr(tmp_new_code, "line_numbers", None)

        matches, deletions, additions = find_line_matches(old_code_string, new_code_str)

        transform_anims = []
        for i, j in matches:
            transform_anims.append(Transform(old_lines[i], new_lines[j]))

        fadeout_anims = []
        for i in deletions:
            fadeout_anims.append(FadeOut(old_lines[i], remover=True))

        fadein_anims = []
        for j in additions:
            fadein_anims.append(FadeIn(new_lines[j]))

        extra_anims = []
        if old_background and new_background:
            extra_anims.append(Transform(old_background, new_background))
        if old_line_numbers and new_line_numbers:
            extra_anims.append(Transform(old_line_numbers, new_line_numbers))

        # if animate codes first, codes covered by background. so background first
        all_anims = []
        if extra_anims:
            all_anims.append(AnimationGroup(*extra_anims))
        if fadeout_anims:
            all_anims.append(AnimationGroup(*fadeout_anims))
        if transform_anims:
            all_anims.append(LaggedStart(*transform_anims, lag_ratio=0.0))
        if fadein_anims:
            all_anims.append(AnimationGroup(*fadein_anims))

        final_group = AnimationGroup(
            *all_anims,
            run_time=run_time,
            rate_func=rate_func,
            lag_ratio=lag_ratio,
        )

        self.code_lines = tmp_new_code.code_lines
        self.line_numbers = new_line_numbers
        self.background = tmp_new_code.background
        self._set_current_code_string(new_code_str)

        return final_group


def find_line_matches(
    old_code_str: str, new_code_str: str
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """line matching algorithm with bruteforce"""
    old_lines = [
        line.lstrip() if line.strip() != "" else None
        for line in old_code_str.splitlines()
    ]
    new_lines = [
        line.lstrip() if line.strip() != "" else None
        for line in new_code_str.splitlines()
    ]

    matches = []
    for i, o_line in enumerate(old_lines):
        if o_line is None:
            continue
        for j, n_line in enumerate(new_lines):
            if n_line is not None and o_line == n_line:
                matches.append((i, j))
                old_lines[i] = None
                new_lines[j] = None
                break

    deletions = [i for i, val in enumerate(old_lines) if val is not None]
    additions = [j for j, val in enumerate(new_lines) if val is not None]

    return matches, deletions, additions
