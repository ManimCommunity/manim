"""Mobject representing highlighted source code listings."""

from __future__ import annotations

__all__ = [
    "Code",
]

import html
import os
import re
from pathlib import Path

import numpy as np
from pygments import highlight
from pygments.formatters.html import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer_for_filename
from pygments.styles import get_all_styles

from manim.constants import *
from manim.mobject.geometry.arc import Dot
from manim.mobject.geometry.polygram import RoundedRectangle
from manim.mobject.geometry.shape_matchers import SurroundingRectangle
from manim.mobject.text.text_mobject import Paragraph
from manim.mobject.types.vectorized_mobject import VGroup
from manim.utils.color import WHITE


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
            style=Code.styles_list[15],
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
         The name of the text font to be used. Defaults to ``"Monospac821 BT"``.
    stroke_width
        Stroke width for text. 0 is recommended, and the default.
    margin
        Inner margin of text from the background. Defaults to 0.3.
    indentation_chars
        "Indentation chars" refers to the spaces/tabs at the beginning of a given code line. Defaults to ``"    "`` (spaces).
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
        (the default), the language will be automatically detected. For the list of
        possible options, visit https://pygments.org/docs/lexers/ and look for
        'aliases or short names'.
    generate_html_file
        Defines whether to generate highlighted html code to the folder `assets/codes/generated_html_files`. Defaults to `False`.

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

    # tuples in the form (name, aliases, filetypes, mimetypes)
    # 'language' is aliases or short names
    # For more information about pygments.lexers visit https://pygments.org/docs/lexers/
    # from pygments.lexers import get_all_lexers
    # all_lexers = get_all_lexers()
    styles_list = list(get_all_styles())
    # For more information about pygments.styles visit https://pygments.org/docs/styles/

    def __init__(
        self,
        file_name: str | os.PathLike | None = None,
        code: str | None = None,
        tab_width: int = 3,
        line_spacing: float = 0.3,
        font_size: float = 24,
        font: str = "Monospac821 BT",
        stroke_width: float = 0,
        margin: float = 0.3,
        indentation_chars: str = "    ",
        background: str = "rectangle",  # or window
        background_stroke_width: float = 1,
        background_stroke_color: str = WHITE,
        corner_radius: float = 0.2,
        insert_line_no: bool = True,
        line_no_from: int = 1,
        line_no_buff: float = 0.4,
        style: str = "vim",
        language: str | None = None,
        generate_html_file: bool = False,
        **kwargs,
    ):
        super().__init__(
            stroke_width=stroke_width,
            **kwargs,
        )
        self.background_stroke_color = background_stroke_color
        self.background_stroke_width = background_stroke_width
        self.tab_width = tab_width
        self.line_spacing = line_spacing
        self.font = font
        self.font_size = font_size
        self.margin = margin
        self.indentation_chars = indentation_chars
        self.background = background
        self.corner_radius = corner_radius
        self.insert_line_no = insert_line_no
        self.line_no_from = line_no_from
        self.line_no_buff = line_no_buff
        self.style = style
        self.language = language
        self.generate_html_file = generate_html_file

        self.file_path = None
        self.file_name = file_name
        if self.file_name:
            self._ensure_valid_file()
            self.code_string = self.file_path.read_text(encoding="utf-8")
        elif code:
            self.code_string = code
        else:
            raise ValueError(
                "Neither a code file nor a code string have been specified.",
            )
        if isinstance(self.style, str):
            self.style = self.style.lower()
        self._gen_html_string()
        strati = self.html_string.find("background:")
        self.background_color = self.html_string[strati + 12 : strati + 19]
        self._gen_code_json()

        self.code = self._gen_colored_lines()
        if self.insert_line_no:
            self.line_numbers = self._gen_line_numbers()
            self.line_numbers.next_to(self.code, direction=LEFT, buff=self.line_no_buff)
        if self.background == "rectangle":
            if self.insert_line_no:
                foreground = VGroup(self.code, self.line_numbers)
            else:
                foreground = self.code
            rect = SurroundingRectangle(
                foreground,
                buff=self.margin,
                color=self.background_color,
                fill_color=self.background_color,
                stroke_width=self.background_stroke_width,
                stroke_color=self.background_stroke_color,
                fill_opacity=1,
            )
            rect.round_corners(self.corner_radius)
            self.background_mobject = rect
        else:
            if self.insert_line_no:
                foreground = VGroup(self.code, self.line_numbers)
            else:
                foreground = self.code
            height = foreground.height + 0.1 * 3 + 2 * self.margin
            width = foreground.width + 0.1 * 3 + 2 * self.margin

            rect = RoundedRectangle(
                corner_radius=self.corner_radius,
                height=height,
                width=width,
                stroke_width=self.background_stroke_width,
                stroke_color=self.background_stroke_color,
                color=self.background_color,
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
                + LEFT * (width / 2 - 0.1 * 5 - self.corner_radius / 2 - 0.05),
            )

            self.background_mobject = VGroup(rect, buttons)
            x = (height - foreground.height) / 2 - 0.1 * 3
            self.background_mobject.shift(foreground.get_center())
            self.background_mobject.shift(UP * x)
        if self.insert_line_no:
            super().__init__(
                self.background_mobject, self.line_numbers, self.code, **kwargs
            )
        else:
            super().__init__(
                self.background_mobject,
                Dot(fill_opacity=0, stroke_opacity=0),
                self.code,
                **kwargs,
            )
        self.move_to(np.array([0, 0, 0]))

    def _ensure_valid_file(self):
        """Function to validate file."""
        if self.file_name is None:
            raise Exception("Must specify file for Code")
        possible_paths = [
            Path() / "assets" / "codes" / self.file_name,
            Path(self.file_name).expanduser(),
        ]
        for path in possible_paths:
            if path.exists():
                self.file_path = path
                return
        error = (
            f"From: {Path.cwd()}, could not find {self.file_name} at either "
            + f"of these locations: {list(map(str, possible_paths))}"
        )
        raise OSError(error)

    def _gen_line_numbers(self):
        """Function to generate line_numbers.

        Returns
        -------
        :class:`~.Paragraph`
            The generated line_numbers according to parameters.
        """
        line_numbers_array = []
        for line_no in range(0, self.code_json.__len__()):
            number = str(self.line_no_from + line_no)
            line_numbers_array.append(number)
        line_numbers = Paragraph(
            *list(line_numbers_array),
            line_spacing=self.line_spacing,
            alignment="right",
            font_size=self.font_size,
            font=self.font,
            disable_ligatures=True,
            stroke_width=self.stroke_width,
        )
        for i in line_numbers:
            i.set_color(self.default_color)
        return line_numbers

    def _gen_colored_lines(self):
        """Function to generate code.

        Returns
        -------
        :class:`~.Paragraph`
            The generated code according to parameters.
        """
        lines_text = []
        for line_no in range(0, self.code_json.__len__()):
            line_str = ""
            for word_index in range(self.code_json[line_no].__len__()):
                line_str = line_str + self.code_json[line_no][word_index][0]
            lines_text.append(self.tab_spaces[line_no] * "\t" + line_str)
        code = Paragraph(
            *list(lines_text),
            line_spacing=self.line_spacing,
            tab_width=self.tab_width,
            font_size=self.font_size,
            font=self.font,
            disable_ligatures=True,
            stroke_width=self.stroke_width,
        )
        for line_no in range(code.__len__()):
            line = code.chars[line_no]
            line_char_index = self.tab_spaces[line_no]
            for word_index in range(self.code_json[line_no].__len__()):
                line[
                    line_char_index : line_char_index
                    + self.code_json[line_no][word_index][0].__len__()
                ].set_color(self.code_json[line_no][word_index][1])
                line_char_index += self.code_json[line_no][word_index][0].__len__()
        return code

    def _gen_html_string(self):
        """Function to generate html string with code highlighted and stores in variable html_string."""
        self.html_string = _hilite_me(
            self.code_string,
            self.language,
            self.style,
            self.insert_line_no,
            "border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;",
            self.file_path,
            self.line_no_from,
        )

        if self.generate_html_file:
            output_folder = Path() / "assets" / "codes" / "generated_html_files"
            output_folder.mkdir(parents=True, exist_ok=True)
            (output_folder / f"{self.file_name}.html").write_text(self.html_string)

    def _gen_code_json(self):
        """Function to background_color, generate code_json and tab_spaces from html_string.
        background_color is just background color of displayed code.
        code_json is 2d array with rows as line numbers
        and columns as a array with length 2 having text and text's color value.
        tab_spaces is 2d array with rows as line numbers
        and columns as corresponding number of indentation_chars in front of that line in code.
        """
        if (
            self.background_color == "#111111"
            or self.background_color == "#272822"
            or self.background_color == "#202020"
            or self.background_color == "#000000"
        ):
            self.default_color = "#ffffff"
        else:
            self.default_color = "#000000"
        # print(self.default_color,self.background_color)
        for i in range(3, -1, -1):
            self.html_string = self.html_string.replace("</" + " " * i, "</")

        # handle pygments bug
        # https://github.com/pygments/pygments/issues/961
        self.html_string = self.html_string.replace("<span></span>", "")

        for i in range(10, -1, -1):
            self.html_string = self.html_string.replace(
                "</span>" + " " * i,
                " " * i + "</span>",
            )
        self.html_string = self.html_string.replace("background-color:", "background:")

        if self.insert_line_no:
            start_point = self.html_string.find("</td><td><pre")
            start_point = start_point + 9
        else:
            start_point = self.html_string.find("<pre")
        self.html_string = self.html_string[start_point:]
        # print(self.html_string)
        lines = self.html_string.split("\n")
        lines = lines[0 : lines.__len__() - 2]
        start_point = lines[0].find(">")
        lines[0] = lines[0][start_point + 1 :]
        # print(lines)
        self.code_json = []
        self.tab_spaces = []
        code_json_line_index = -1
        for line_index in range(0, lines.__len__()):
            # print(lines[line_index])
            self.code_json.append([])
            code_json_line_index = code_json_line_index + 1
            if lines[line_index].startswith(self.indentation_chars):
                start_point = lines[line_index].find("<")
                starting_string = lines[line_index][:start_point]
                indentation_chars_count = lines[line_index][:start_point].count(
                    self.indentation_chars,
                )
                if (
                    starting_string.__len__()
                    != indentation_chars_count * self.indentation_chars.__len__()
                ):
                    lines[line_index] = (
                        "\t" * indentation_chars_count
                        + starting_string[
                            starting_string.rfind(self.indentation_chars)
                            + self.indentation_chars.__len__() :
                        ]
                        + lines[line_index][start_point:]
                    )
                else:
                    lines[line_index] = (
                        "\t" * indentation_chars_count + lines[line_index][start_point:]
                    )
            indentation_chars_count = 0
            if lines[line_index]:
                while lines[line_index][indentation_chars_count] == "\t":
                    indentation_chars_count = indentation_chars_count + 1
            self.tab_spaces.append(indentation_chars_count)
            # print(lines[line_index])
            lines[line_index] = self._correct_non_span(lines[line_index])
            # print(lines[line_index])
            words = lines[line_index].split("<span")
            for word_index in range(1, words.__len__()):
                color_index = words[word_index].find("color:")
                if color_index == -1:
                    color = self.default_color
                else:
                    starti = words[word_index][color_index:].find("#")
                    color = words[word_index][
                        color_index + starti : color_index + starti + 7
                    ]
                start_point = words[word_index].find(">")
                end_point = words[word_index].find("</span>")
                text = words[word_index][start_point + 1 : end_point]
                text = html.unescape(text)
                if text != "":
                    # print(text, "'" + color + "'")
                    self.code_json[code_json_line_index].append([text, color])
        # print(self.code_json)

    def _correct_non_span(self, line_str: str):
        """Function put text color to those strings that don't have one according to background_color of displayed code.

        Parameters
        ---------
        line_str
            Takes a html element's string to put color to it according to background_color of displayed code.

        Returns
        -------
        :class:`str`
            The generated html element's string with having color attributes.
        """
        words = line_str.split("</span>")
        line_str = ""
        for i in range(0, words.__len__()):
            if i != words.__len__() - 1:
                j = words[i].find("<span")
            else:
                j = words[i].__len__()
            temp = ""
            starti = -1
            for k in range(0, j):
                if words[i][k] == "\t" and starti == -1:
                    continue
                else:
                    if starti == -1:
                        starti = k
                    temp = temp + words[i][k]
            if temp != "":
                if i != words.__len__() - 1:
                    temp = (
                        '<span style="color:'
                        + self.default_color
                        + '">'
                        + words[i][starti:j]
                        + "</span>"
                    )
                else:
                    temp = (
                        '<span style="color:'
                        + self.default_color
                        + '">'
                        + words[i][starti:j]
                    )
                temp = temp + words[i][j:]
                words[i] = temp
            if words[i] != "":
                line_str = line_str + words[i] + "</span>"
        return line_str


def _hilite_me(
    code: str,
    language: str,
    style: str,
    insert_line_no: bool,
    divstyles: str,
    file_path: Path,
    line_no_from: int,
):
    """Function to highlight code from string to html.

    Parameters
    ---------
    code
        Code string.
    language
        The name of the programming language the given code was written in.
    style
        Code style name.
    insert_line_no
        Defines whether line numbers should be inserted in the html file.
    divstyles
        Some html css styles.
    file_path
        Path of code file.
    line_no_from
        Defines the first line's number in the line count.
    """
    style = style or "colorful"
    defstyles = "overflow:auto;width:auto;"

    formatter = HtmlFormatter(
        style=style,
        linenos=False,
        noclasses=True,
        cssclass="",
        cssstyles=defstyles + divstyles,
        prestyles="margin: 0",
    )
    if language is None and file_path:
        lexer = guess_lexer_for_filename(file_path, code)
        html = highlight(code, lexer, formatter)
    elif language is None:
        raise ValueError(
            "The code language has to be specified when rendering a code string",
        )
    else:
        html = highlight(code, get_lexer_by_name(language, **{}), formatter)
    if insert_line_no:
        html = _insert_line_numbers_in_html(html, line_no_from)
    html = "<!-- HTML generated by Code() -->" + html
    return html


def _insert_line_numbers_in_html(html: str, line_no_from: int):
    """Function that inserts line numbers in the highlighted HTML code.

    Parameters
    ---------
    html
        html string of highlighted code.
    line_no_from
        Defines the first line's number in the line count.

    Returns
    -------
    :class:`str`
        The generated html string with having line numbers.
    """
    match = re.search("(<pre[^>]*>)(.*)(</pre>)", html, re.DOTALL)
    if not match:
        return html
    pre_open = match.group(1)
    pre = match.group(2)
    pre_close = match.group(3)

    html = html.replace(pre_close, "</pre></td></tr></table>")
    numbers = range(line_no_from, line_no_from + pre.count("\n") + 1)
    format_lines = "%" + str(len(str(numbers[-1]))) + "i"
    lines = "\n".join(format_lines % i for i in numbers)
    html = html.replace(
        pre_open,
        "<table><tr><td>" + pre_open + lines + "</pre></td><td>" + pre_open,
    )
    return html
