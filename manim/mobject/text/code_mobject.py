"""Mobject representing highlighted source code listings."""

from __future__ import annotations

__all__ = [
    "Code",
]

import os
from pathlib import Path

from pygments import styles
from pygments.lexer import Lexer
from pygments.lexers import (
    get_lexer_by_name,
    get_lexer_for_filename,
    guess_lexer,
)
from pygments.token import _TokenType
from pygments.util import ClassNotFound

# from pygments.styles import get_all_styles
from manim import logger
from manim.constants import *
from manim.mobject.geometry.arc import Dot
from manim.mobject.geometry.polygram import RoundedRectangle
from manim.mobject.geometry.shape_matchers import SurroundingRectangle
from manim.mobject.text.text_mobject import Paragraph
from manim.mobject.types.vectorized_mobject import VGroup
from manim.utils.color import WHITE


class CodeColorFormatter:
    """Simple Formatter which is based of `Pygments.Formatter`. Formatter outputs text-color mapping in format: `list[tuple[text:str, color:str]]`
    Class bypasses all normal pygments `Formatter` protocols and outputs and uses only needed functions for efficiency.
    This works only in context of :class:``Code``"""

    DEFAULT_STYLE = "vim"

    def __init__(
        self,
        style,
        code,
        language,
        file_name,
    ):
        self.code = code
        self.lexer: Lexer = self.find_lexer(file_name, code, language)

        self.style = self.get_style(style)

        self.bg_color = self.style.background_color
        self.default_color = self.opposite_color(self.bg_color)
        self.styles: dict[_TokenType, str] = {}
        """`dict[tokentype, color]` mapping"""

        for token, style in self.style:
            value = style["color"]
            if not value:
                value = self.default_color
            self.styles[token] = value

    def format_code(self):
        def add_to_mapping(word: str, token: _TokenType):
            self.mapping[-1].append((word, "#" + self.styles[token]))

        def style_token_support(token: _TokenType):
            if token not in self.styles:
                # Tokens are stored in hierarchy:
                # e.g. Token.Literal.String.Douple parent is:
                # Token.Literal.String
                return token.parent
            else:
                return token

        self.mapping: list[tuple[str, str]] = [[]]
        self.tokens = self.lexer.get_tokens(self.code)
        lasttype, lastval = next(self.tokens)

        for token_type, value in self.tokens:
            token_type = style_token_support(token_type)

            if "\n" in lastval:
                if len(lastval) > 1:
                    # There is cases in which Tokeniser returns values with newline attached into.
                    culled = lastval.removesuffix("\n")
                    add_to_mapping(culled, lasttype)

                self.mapping.append([])
                lastval = value
                lasttype = token_type

            elif value == " " or self.styles[token_type] == self.styles[lasttype]:
                # NOTE This is hack for later efficiency, will broke token hierarchy if other style coding checks is added in future
                lastval += value

            else:
                add_to_mapping(lastval, lasttype)
                lastval = value
                lasttype = token_type

        return self.mapping

    def get_colors(self):
        return self.bg_color, self.default_color

    @staticmethod
    def opposite_color(color: str) -> str:
        """Generate opposite color string"""
        if color == "#000000":
            return "#ffffff"
        elif color == "#ffffff":
            return "#000000"
        else:
            new_hexes = []

            for i in range(1, 6, 2):
                hex_str = color[i : i + 2]
                hex_int = int(hex_str, 16)
                new_hex = hex(abs(hex_int - 255)).strip("0x")
                new_hex = new_hex if len(new_hex) > 1 else "0" + new_hex
                new_hexes.append(new_hex)
            return "#" + "".join(new_hexes)

    @staticmethod
    def find_lexer(file_name: str, code: str, language) -> Lexer:
        try:
            if language:
                lexer = get_lexer_by_name(language)
            elif file_name:
                lexer = get_lexer_for_filename(file_name, code)
            elif code:
                lexer = guess_lexer(code)

            return lexer

        except ClassNotFound as a:
            a.add_note(
                f"Could not resolve pygments lexer for a Code object. File:{file_name}, Code: {code}, language: {language}"
            )
            raise a

    @classmethod
    def get_style(cls, style: str | any) -> bool:
        try:
            if isinstance(style, str):
                style = style.lower()
                style = styles.get_style_by_name(style)
                return style
            else:
                raise TypeError

        except ClassNotFound:
            logger.warning(
                f'{Code.__name__}: style "{style}" is not supported, using default style: "{cls.DEFAULT_STYLE}" '
            )
            return styles.get_style_by_name(cls.DEFAULT_STYLE)
        except TypeError:
            logger.warning(
                f'Style should be a string type. Used value {style} is type of {type(style)}. Using default type "{cls.DEFAULT_STYLE}"  '
            )
            return styles.get_style_by_name(cls.DEFAULT_STYLE)


class Code(VGroup):
    """A highlighted source code listing.

    An object ``listing`` of :class:`.Code` is a :class:`.VGroup` consisting
    of three objects:

    - The background, ``listing.background_mobject``. This is either
      a :class:`.Rectangle` (if the listing has been initialized with
      ``background="rectangle"``, the default option) or a :class:`.VGroup`
      resembling a window (if ``background="window"`` has been passed).

    - The line numbers, ``listing.line_numbers`` (a :class:`.Paragraph`
      object).

    - The highlighted code itself, ``listing.code`` (a :class:`.Paragraph`
      object).

    .. WARNING::

        Using a :class:`.Transform` on text with leading whitespace (and in
        this particular case: code) can look
        `weird <https://github.com/3b1b/manim/issues/1067>`_. Consider using
        :meth:`remove_invisible_chars` to resolve this issue.

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

    We can also render code passed as a string (but note that
    the language has to be specified in this case):

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
    file_name
        Name of the code file to display.
    code
        If ``file_name`` is not specified, a code string can be
        passed directly.
    tab_width
        Number of space characters corresponding to a tab character. Defaults to 3.
    line_spacing
        Amount of space between lines in relation to font size. Defaults to 0.3, which means 30% of font size.
    font_size
        A number which scales displayed code. Defaults to 24.
    font
        The name of the text font to be used. Defaults to ``"Monospace"``.
        This is either a system font or one loaded with `text.register_font()`. Note
        that font family names may be different across operating systems.
    stroke_width
        Stroke width for text. 0 is recommended, and the default.
    margin
        Inner margin of text from the background. Defaults to 0.3.
    background
        Defines the background's type. Currently supports only ``"rectangle"`` (default) and ``"window"``.
    background_stroke_width
        Defines the stroke width of the background. Defaults to 1.
    background_stroke_color
        Defines the stroke color for the background. Defaults to ``WHITE``.
    corner_radius
        Defines the corner radius for the background. Defaults to 0.2.
    insert_line_no
        Defines whether line numbers should be inserted in displayed code. Defaults to ``True``.
    line_no_from
        Defines the first line's number in the line count. Defaults to 1.
    line_no_buff
        Defines the spacing between line numbers and displayed code. Defaults to 0.4.
    style
        Defines the style type of displayed code. You can see possible names of styles in with :attr:`styles_list`. Defaults to ``"vim"``.
    language
        Specifies the programming language the given code was written in. If ``None``
        (the default), the language is tried to detect from code. For the list of
        possible options, visit https://pygments.org/docs/lexers/ and look for
        'aliases or short names'.
    warn_missing_font
        If True (default), Manim will issue a warning if the font does not exist in the
        (case-sensitive) list of fonts returned from `manimpango.list_fonts()`.

    Attributes
    ----------
    background_mobject : :class:`~.VGroup`
        The background of the code listing.
    line_numbers : :class:`~.Paragraph`
        The line numbers for the code listing. Empty, if
        ``insert_line_no=False`` has been specified.
    code : :class:`~.Paragraph`
        The highlighted code.
    """

    _styles_list_cache = None
    """Containing all pygments supported styles."""

    def __init__(
        self,
        file_name: str | os.PathLike | None = None,
        code: str | None = None,
        language: str | None = None,
        tab_width: int = 3,
        line_spacing: float = 0.8,
        font_size: float = 24,
        font: str = "Monospace",
        stroke_width: float = 0,
        margin: float = 0.3,
        background: str = "neutral",
        background_stroke_width: float = 1,
        background_stroke_color: str = WHITE,
        corner_radius: float = 0.2,
        insert_line_no: bool = True,
        line_no_from: int = 1,
        line_no_buff: float = 0.4,
        style: str = "vim",
        generate_html_file: bool = False,
        warn_missing_font: bool = True,
        **kwargs,
    ):
        if generate_html_file:
            logger.warning(
                f"{Code.__name__} argument 'generate_html_file' is deprecated and does not work anymore"
            )

        code_string = create_code_string(file_name, code)

        formatter = CodeColorFormatter(style, code_string, language, file_name)
        mapping = formatter.format_code()
        bg_color, default_color = formatter.get_colors()

        self.code = ColoredCodeText(
            stroke_width,
            mapping,
            line_spacing,
            tab_width,
            font_size,
            font,
            default_color,
            warn_missing_font,
        )

        if insert_line_no:
            self.line_numbers: Paragraph = LineNumbers(
                line_no_from,
                len(mapping),
                default_color,
                stroke_width,
                line_spacing,
                font_size,
                font,
                warn_missing_font,
            )

            self.line_numbers.next_to(self.code, direction=LEFT, buff=line_no_buff)
            foreground = VGroup(self.code, self.line_numbers)
        else:
            foreground = self.code

        bg_function = (
            NeutralStyle if background in ["rectangle", "neutral"] else MacOsStyle
        )

        self.background_mobject = bg_function(
            foreground,
            margin,
            bg_color,
            background_stroke_width,
            background_stroke_color,
            corner_radius,
        )

        return super().__init__(
            self.background_mobject,
            foreground,
            stroke_width=stroke_width,
            **kwargs,
        )

    @classmethod
    def get_styles_list(cls) -> list[str]:
        """Return a list of available code styles.
        For more information about pygments.styles visit https://pygments.org/docs/styles/
        """
        if cls._styles_list_cache is None:
            cls._styles_list_cache = list(styles.get_all_styles())
        return cls._styles_list_cache


def create_code_string(file_name: str | Path, code: str) -> str:
    def _search_file_path(path_name: Path | str):
        """Function to search and find the code file"""
        # TODO Hard coded directories
        possible_paths = [
            Path() / "assets" / "codes" / path_name,
            Path(path_name).expanduser(),
        ]

        for path in possible_paths:
            if path.exists():
                return path
        else:
            raise FileNotFoundError(
                f"@ {Path.cwd()}: {Code.__name__} Couldn't find code file from these paths: {possible_paths}"
            )

    if file_name:
        assert isinstance(file_name, (str, Path))
        file_path = _search_file_path(file_name)
        return file_path.read_text(encoding="utf-8")

    elif code:
        assert isinstance(code, str)
        return code
    else:
        raise ValueError(
            "Neither a code file nor a code string has been specified. Cannot generate Code block",
        )


# Mobject constructors:


def LineNumbers(
    starting_no,
    lines,
    default_color,
    line_width,
    line_spacing,
    font_size,
    font,
    warn_missing_font,
) -> Paragraph:
    """Function generates line_numbers ``Paragraph`` mobject."""
    line_no_strings = [str(i) for i in range(starting_no, lines + starting_no)]

    line_numbers: Paragraph = Paragraph(
        *line_no_strings,
        line_spacing=line_spacing,
        alignment="right",
        font_size=font_size,
        font=font,
        disable_ligatures=True,
        stroke_width=line_width,
        warn_missing_font=warn_missing_font,
    )
    for i in line_numbers:
        i.set_color(default_color)

    return line_numbers


def ColoredCodeText(
    line_width,
    text_mapping: list[list[str, str]],
    line_spacing,
    tab_width,
    font_size,
    font,
    default_color,
    warn_missing_font=False,
) -> Paragraph:
    """Function generates code-block with code coloration``Paragraph`` mobject"""
    lines_text = []

    for line in text_mapping:
        line_str = ""
        for style_map in line:
            line_str += style_map[0]
        lines_text.append(line_str)

    code_mobject = Paragraph(
        *list(lines_text),
        line_spacing=line_spacing,
        tab_width=tab_width,
        font_size=font_size,
        font=font,
        disable_ligatures=True,
        stroke_width=line_width,
        warn_missing_font=warn_missing_font,
        stroke_color=default_color,
    )

    try:
        mobject_lines = len(code_mobject)
        mapping_lines = len(text_mapping)
        assert mobject_lines == mapping_lines

        for line_no in range(mobject_lines):
            line = code_mobject.chars[line_no]
            line_char_index = 0
            line_length = len(text_mapping[line_no])

            for word_index in range(line_length):
                word_mapping = text_mapping[line_no][word_index]
                word_length = len(word_mapping[0])
                color = word_mapping[1]
                line[line_char_index : line_char_index + word_length].set_color(color)
                line_char_index += word_length

    except AssertionError as a:
        from pygments import __version__

        error = (
            "ERROR: While parsing a Code object there was an error: lines of Paraghraph Mobject does not match with mapped lines.\n"
            + "This most likely happened due to an unknown bug in parser)\n"
            + f"pygments version: {__version__}\n"
            "Result: Could not stylize code block properly"
        )
        logger.error(error)
        a.add_note(error)
        if logger.level == 10:  # Debugging == 10
            raise a

    return code_mobject


def NeutralStyle(
    foreground: VGroup | Paragraph,
    margin,
    bg_col,
    bg_stroke_width,
    bg_stroke_color,
    corner_radius,
) -> SurroundingRectangle:
    rect = SurroundingRectangle(
        foreground,
        buff=margin,
        color=bg_col,
        fill_color=bg_col,
        stroke_width=bg_stroke_width,
        stroke_color=bg_stroke_color,
        fill_opacity=1,
    )
    rect.round_corners(corner_radius)
    return rect


def MacOsStyle(
    foreground: VGroup | Paragraph,
    margin,
    bg_col,
    bg_stroke_width,
    bg_stroke_color,
    corner_radius,
) -> VGroup:
    height = foreground.height + 0.1 * 3 + 2 * margin
    width = foreground.width + 0.1 * 3 + 2 * margin

    rect = RoundedRectangle(
        corner_radius=corner_radius,
        height=height,
        width=width,
        stroke_width=bg_stroke_width,
        stroke_color=bg_stroke_color,
        color=bg_col,
        fill_opacity=1,
    )
    red_button = Dot(radius=0.1, stroke_width=0, color="#ff5f56")
    red_button.shift(LEFT * 0.1 * 3)
    yellow_button = Dot(radius=0.1, stroke_width=0, color="#ffbd2e")
    green_button = Dot(radius=0.1, stroke_width=0, color="#27c93f")
    green_button.shift(RIGHT * 0.1 * 3)
    buttons = VGroup(red_button, yellow_button, green_button)
    buttons.shift(
        UP * (height / 2 - 0.1 * 2 - 0.05)
        + LEFT * (width / 2 - 0.1 * 5 - corner_radius / 2 - 0.05),
    )

    window = VGroup(rect, buttons)
    x = (height - foreground.height) / 2 - 0.1 * 3
    window.shift(foreground.get_center())
    window.shift(UP * x)
    return window
