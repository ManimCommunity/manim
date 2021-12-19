"""Mobjects used for displaying (non-LaTeX) text.

The simplest way to add text to your animations is to use the :class:`~.Text` class. It uses the Pango library to render text.
With Pango, you are also able to render non-English alphabets like `你好` or  `こんにちは` or `안녕하세요` or `مرحبا بالعالم`.

Examples
--------

.. manim:: HelloWorld
    :save_last_frame:

    class HelloWorld(Scene):
        def construct(self):
            text = Text('Hello world').scale(3)
            self.add(text)

.. manim:: TextAlignment
    :save_last_frame:

    class TextAlignment(Scene):
        def construct(self):
            title = Text("K-means clustering and Logistic Regression", color=WHITE)
            title.scale(0.75)
            self.add(title.to_edge(UP))

            t1 = Text("1. Measuring").set_color(WHITE)
            t1.next_to(ORIGIN, direction=RIGHT, aligned_edge=UP)

            t2 = Text("2. Clustering").set_color(WHITE)
            t2.next_to(t1, direction=DOWN, aligned_edge=LEFT)

            t3 = Text("3. Regression").set_color(WHITE)
            t3.next_to(t2, direction=DOWN, aligned_edge=LEFT)

            t4 = Text("4. Prediction").set_color(WHITE)
            t4.next_to(t3, direction=DOWN, aligned_edge=LEFT)

            x = VGroup(t1, t2, t3, t4).scale(0.7)
            x.set_opacity(0.5)
            x.submobjects[1].set_opacity(1)
            self.add(x)

"""

# __all__ = ["Text", "Paragraph", "CairoText", "MarkupText", "register_font"]


import copy
import hashlib
import os
import re
import typing
from contextlib import contextmanager
from pathlib import Path

import manimpango
from manimpango import MarkupUtils, PangoUtils, TextSetting

from ... import config, logger
from ...constants import *
from ...mobject.opengl_geometry import OpenGLDot
from ...mobject.svg.opengl_svg_mobject import OpenGLSVGMobject
from ...mobject.types.opengl_vectorized_mobject import OpenGLVGroup
from ...utils.color import WHITE, Colors

TEXT_MOB_SCALE_FACTOR = 0.05


def remove_invisible_chars(mobject):
    """Function to remove unwanted invisible characters from some mobject

    Parameters
    ----------
    mobject : :class:`~.SVGMobject`
        Any SVGMobject from which we want to remove unwanted invisible characters.

    Returns
    -------
    :class:`~.SVGMobject`
        The SVGMobject without unwanted invisible characters.
    """

    iscode = False
    if mobject.__class__.__name__ == "Text":
        mobject = mobject[:]
    elif mobject.__class__.__name__ == "Code":
        iscode = True
        code = mobject
        mobject = mobject.code
    mobject_without_dots = VGroup()
    if mobject[0].__class__ == VGroup:
        for i in range(mobject.__len__()):
            mobject_without_dots.add(VGroup())
            mobject_without_dots[i].add(*(k for k in mobject[i] if k.__class__ != Dot))
    else:
        mobject_without_dots.add(*(k for k in mobject if k.__class__ != Dot))
    if iscode:
        code.code = mobject_without_dots
        return code
    return mobject_without_dots


class OpenGLParagraph(OpenGLVGroup):
    r"""Display a paragraph of text.

    For a given :class:`.Paragraph` ``par``, the attribute ``par.chars`` is a
    :class:`.VGroup` containing all the lines. In this context, every line is
    constructed as a :class:`.VGroup` of characters contained in the line.


    Parameters
    ----------
    line_spacing : :class:`int`, optional
        Represents the spacing between lines. Default to -1, which means auto.
    alignment : :class:`str`, optional
        Defines the alignment of paragraph. Default to "left". Possible values are "left", "right", "center"

    Examples
    --------
    Normal usage::

        paragraph = Paragraph('this is a awesome', 'paragraph',
                              'With \nNewlines', '\tWith Tabs',
                              '  With Spaces', 'With Alignments',
                              'center', 'left', 'right')

    Remove unwanted invisible characters::

        self.play(Transform(remove_invisible_chars(paragraph.chars[0:2]),
                            remove_invisible_chars(paragraph.chars[3][0:3]))

    """

    def __init__(self, *text, line_spacing=-1, alignment=None, **config):
        self.line_spacing = line_spacing
        self.alignment = alignment

        lines_str = "\n".join(list(text))
        self.lines_text = OpenGLText(lines_str, line_spacing=line_spacing, **config)
        lines_str_list = lines_str.split("\n")
        self.chars = self.gen_chars(lines_str_list)

        chars_lines_text_list = OpenGLVGroup()
        char_index_counter = 0
        for line_index in range(lines_str_list.__len__()):
            chars_lines_text_list.add(
                self.lines_text[
                    char_index_counter : char_index_counter
                    + lines_str_list[line_index].__len__()
                    + 1
                ],
            )
            char_index_counter += lines_str_list[line_index].__len__() + 1
        self.lines = []
        self.lines.append([])
        for line_no in range(chars_lines_text_list.__len__()):
            self.lines[0].append(chars_lines_text_list[line_no])
        self.lines_initial_positions = []
        for line_no in range(self.lines[0].__len__()):
            self.lines_initial_positions.append(self.lines[0][line_no].get_center())
        self.lines.append([])
        self.lines[1].extend(
            [self.alignment for _ in range(chars_lines_text_list.__len__())],
        )
        super().__init__(
            *(self.lines[0][i] for i in range(self.lines[0].__len__())), **config
        )
        self.move_to(np.array([0, 0, 0]))
        if self.alignment:
            self.set_all_lines_alignments(self.alignment)

    def gen_chars(self, lines_str_list):
        """Function to convert plain string to 2d-VGroup of chars. 2d-VGroup mean "VGroup of VGroup".

        Parameters
        ----------
        lines_str_list : :class:`str`
            Plain text string.

        Returns
        -------
        :class:`~.VGroup`
            The generated 2d-VGroup of chars.
        """
        char_index_counter = 0
        chars = OpenGLVGroup()
        for line_no in range(lines_str_list.__len__()):
            chars.add(OpenGLVGroup())
            chars[line_no].add(
                *self.lines_text.chars[
                    char_index_counter : char_index_counter
                    + lines_str_list[line_no].__len__()
                    + 1
                ]
            )
            char_index_counter += lines_str_list[line_no].__len__() + 1
        return chars

    def set_all_lines_alignments(self, alignment):
        """Function to set all line's alignment to a specific value.

        Parameters
        ----------
        alignment : :class:`str`
            Defines the alignment of paragraph. Possible values are "left", "right", "center".
        """
        for line_no in range(0, self.lines[0].__len__()):
            self.change_alignment_for_a_line(alignment, line_no)
        return self

    def set_line_alignment(self, alignment, line_no):
        """Function to set one line's alignment to a specific value.

        Parameters
        ----------
        alignment : :class:`str`
            Defines the alignment of paragraph. Possible values are "left", "right", "center".
        line_no : :class:`int`
            Defines the line number for which we want to set given alignment.
        """
        self.change_alignment_for_a_line(alignment, line_no)
        return self

    def set_all_lines_to_initial_positions(self):
        """Set all lines to their initial positions."""
        self.lines[1] = [None for _ in range(self.lines[0].__len__())]
        for line_no in range(0, self.lines[0].__len__()):
            self[line_no].move_to(
                self.get_center() + self.lines_initial_positions[line_no],
            )
        return self

    def set_line_to_initial_position(self, line_no):
        """Function to set one line to initial positions.

        Parameters
        ----------
        line_no : :class:`int`
            Defines the line number for which we want to set given alignment.
        """
        self.lines[1][line_no] = None
        self[line_no].move_to(self.get_center() + self.lines_initial_positions[line_no])
        return self

    def change_alignment_for_a_line(self, alignment, line_no):
        """Function to change one line's alignment to a specific value.

        Parameters
        ----------
        alignment : :class:`str`
            Defines the alignment of paragraph. Possible values are "left", "right", "center".
        line_no : :class:`int`
            Defines the line number for which we want to set given alignment.
        """
        self.lines[1][line_no] = alignment
        if self.lines[1][line_no] == "center":
            self[line_no].move_to(
                np.array([self.get_center()[0], self[line_no].get_center()[1], 0]),
            )
        elif self.lines[1][line_no] == "right":
            self[line_no].move_to(
                np.array(
                    [
                        self.get_right()[0] - self[line_no].width / 2,
                        self[line_no].get_center()[1],
                        0,
                    ],
                ),
            )
        elif self.lines[1][line_no] == "left":
            self[line_no].move_to(
                np.array(
                    [
                        self.get_left()[0] + self[line_no].width / 2,
                        self[line_no].get_center()[1],
                        0,
                    ],
                ),
            )


class OpenGLText(OpenGLSVGMobject):
    r"""Display (non-LaTeX) text rendered using `Pango <https://pango.gnome.org/>`_.

    Text objects behave like a :class:`.VGroup`-like iterable of all characters
    in the given text. In particular, slicing is possible.

    Parameters
    ----------
    text : :class:`str`
        The text that need to created as mobject.

    Returns
    -------
    :class:`Text`
        The mobject like :class:`.VGroup`.

    Examples
    ---------

    .. manim:: Example1Text
        :save_last_frame:

        class Example1Text(Scene):
            def construct(self):
                text = Text('Hello world').scale(3)
                self.add(text)

    .. manim:: TextColorExample
        :save_last_frame:

        class TextColorExample(Scene):
            def construct(self):
                text1 = Text('Hello world', color=BLUE).scale(3)
                text2 = Text('Hello world', gradient=(BLUE, GREEN)).scale(3).next_to(text1, DOWN)
                self.add(text1, text2)

    .. manim:: TextItalicAndBoldExample
        :save_last_frame:

        class TextItalicAndBoldExample(Scene):
            def construct(self):
                text0 = Text('Hello world', slant=ITALIC)
                text1 = Text('Hello world', t2s={'world':ITALIC})
                text2 = Text('Hello world', weight=BOLD)
                text3 = Text('Hello world', t2w={'world':BOLD})
                self.add(text0,text1, text2,text3)
                for i,mobj in enumerate(self.mobjects):
                    mobj.shift(DOWN*(i-1))


    .. manim:: TextMoreCustomization
            :save_last_frame:

            class TextMoreCustomization(Scene):
                def construct(self):
                    text1 = Text(
                        'Google',
                        t2c={'[:1]': '#3174f0', '[1:2]': '#e53125',
                             '[2:3]': '#fbb003', '[3:4]': '#3174f0',
                             '[4:5]': '#269a43', '[5:]': '#e53125'}, size=1.2).scale(3)
                    self.add(text1)

    As :class:`Text` uses Pango to render text, rendering non-English
    characters is easily possible:

    .. manim:: MultipleFonts
        :save_last_frame:

        class MultipleFonts(Scene):
            def construct(self):
                morning = Text("வணக்கம்", font="sans-serif")
                japanese = Text(
                    "見 角 言 谷  辛 辰 辵 邑 酉 釆 里!", t2c={"見 角 言": BLUE}
                )  # works same as ``Text``.
                mess = Text("Multi-Language", style=BOLD)
                russ = Text("Здравствуйте मस नम म ", font="sans-serif")
                hin = Text("नमस्ते", font="sans-serif")
                arb = Text(
                    "صباح الخير \n تشرفت بمقابلتك", font="sans-serif"
                )  # don't mix RTL and LTR languages nothing shows up then ;-)
                chinese = Text("臂猿「黛比」帶著孩子", font="sans-serif")
                self.add(morning, japanese, mess, russ, hin, arb, chinese)
                for i,mobj in enumerate(self.mobjects):
                    mobj.shift(DOWN*(i-3))


    .. manim:: PangoRender
        :quality: low

        class PangoRender(Scene):
            def construct(self):
                morning = Text("வணக்கம்", font="sans-serif")
                self.play(Write(morning))
                self.wait(2)

    Tests
    -----

    Check that the creation of :class:`~.Text` works::

        >>> Text('The horse does not eat cucumber salad.')
        Text('The horse does not eat cucumber salad.')

    """

    def __init__(
        self,
        text: str,
        fill_opacity: float = 1.0,
        stroke_width: int = 0,
        color: str = WHITE,
        size: int = 1,
        line_spacing: int = -1,
        font: str = "",
        slant: str = NORMAL,
        weight: str = NORMAL,
        t2c: typing.Dict[str, str] = None,
        t2f: typing.Dict[str, str] = None,
        t2g: typing.Dict[str, tuple] = None,
        t2s: typing.Dict[str, str] = None,
        t2w: typing.Dict[str, str] = None,
        gradient: tuple = None,
        tab_width: int = 4,
        # Mobject
        height: int = None,
        width: int = None,
        should_center: bool = True,
        unpack_groups: bool = True,
        disable_ligatures: bool = False,
        **kwargs,
    ):
        self.size = size
        self.line_spacing = line_spacing
        self.font = font
        self.slant = slant
        self.weight = weight
        self.gradient = gradient
        self.tab_width = tab_width
        if t2c is None:
            t2c = {}
        if t2f is None:
            t2f = {}
        if t2g is None:
            t2g = {}
        if t2s is None:
            t2s = {}
        if t2w is None:
            t2w = {}
        # If long form arguments are present, they take precedence
        t2c = kwargs.pop("text2color", t2c)
        t2f = kwargs.pop("text2font", t2f)
        t2g = kwargs.pop("text2gradient", t2g)
        t2s = kwargs.pop("text2slant", t2s)
        t2w = kwargs.pop("text2weight", t2w)
        self.t2c = t2c
        self.t2f = t2f
        self.t2g = t2g
        self.t2s = t2s
        self.t2w = t2w

        self.original_text = text
        self.disable_ligatures = disable_ligatures
        text_without_tabs = text
        if text.find("\t") != -1:
            text_without_tabs = text.replace("\t", " " * self.tab_width)
        self.text = text_without_tabs
        if self.line_spacing == -1:
            self.line_spacing = self.size + self.size * 0.3
        else:
            self.line_spacing = self.size + self.size * self.line_spacing
        file_name = self.text2svg()
        PangoUtils.remove_last_M(file_name)
        super().__init__(
            file_name,
            color=color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            height=height,
            width=width,
            should_center=should_center,
            unpack_groups=unpack_groups,
            **kwargs,
        )
        self.text = text
        if self.disable_ligatures:
            self.submobjects = [*self.gen_chars()]
        self.chars = OpenGLVGroup(*self.submobjects)
        self.text = text_without_tabs.replace(" ", "").replace("\n", "")
        if config["renderer"] == "opengl":
            nppc = self.n_points_per_curve
        else:
            nppc = self.n_points_per_cubic_curve
        for each in self:
            if config["renderer"] == "opengl":
                points = each.data["points"]
            else:
                nppc = each.points
            if len(points) == 0:
                continue
            last = points[0]
            each.clear_points()
            for index, point in enumerate(points):
                each.append_points([point])
                if (
                    index != len(points) - 1
                    and (index + 1) % nppc == 0
                    and any(point != points[index + 1])
                ):
                    each.add_line_to(last)
                    last = points[index + 1]
            each.add_line_to(last)
        if self.t2c:
            self.set_color_by_t2c()
        if self.gradient:
            self.set_color_by_gradient(*self.gradient)
        if self.t2g:
            self.set_color_by_t2g()
        # anti-aliasing
        if height is None and width is None:
            self.scale(TEXT_MOB_SCALE_FACTOR)

    def __repr__(self):
        return f"Text({repr(self.original_text)})"

    def gen_chars(self):
        chars = OpenGLVGroup()
        submobjects_char_index = 0
        for char_index in range(self.text.__len__()):
            if self.text[char_index] in (" ", "\t", "\n"):
                space = OpenGLDot(radius=0, fill_opacity=0, stroke_opacity=0)
                if char_index == 0:
                    space.move_to(self.submobjects[submobjects_char_index].get_center())
                else:
                    space.move_to(
                        self.submobjects[submobjects_char_index - 1].get_center(),
                    )
                chars.add(space)
            else:
                chars.add(self.submobjects[submobjects_char_index])
                submobjects_char_index += 1
        return chars

    def find_indexes(self, word: str, text: str):
        """Internally used function. Finds the indexes of ``text`` in ``word``."""
        temp = re.match(r"\[([0-9\-]{0,}):([0-9\-]{0,})\]", word)
        if temp:
            start = int(temp.group(1)) if temp.group(1) != "" else 0
            end = int(temp.group(2)) if temp.group(2) != "" else len(text)
            start = len(text) + start if start < 0 else start
            end = len(text) + end if end < 0 else end
            return [(start, end)]
        indexes = []
        index = text.find(word)
        while index != -1:
            indexes.append((index, index + len(word)))
            index = text.find(word, index + len(word))
        return indexes

    # def full2short(self, kwargs):
    #     """Internally used function. Formats some expansion to short forms.
    #     text2color -> t2c
    #     text2font -> t2f
    #     text2gradient -> t2g
    #     text2slant -> t2s
    #     text2weight -> t2w
    #     """
    #     if "text2color" in kwargs:
    #         self.t2c = kwargs.pop("text2color")
    #     if "text2font" in kwargs:
    #         self.t2f = kwargs.pop("text2font")
    #     if "text2gradient" in kwargs:
    #         self.t2g = kwargs.pop("text2gradient")
    #     if "text2slant" in kwargs:
    #         self.t2s = kwargs.pop("text2slant")
    #     if "text2weight" in kwargs:
    #         self.t2w = kwargs.pop("text2weight")

    def set_color_by_t2c(self, t2c=None):
        """Internally used function. Sets colour for specified strings."""
        t2c = t2c if t2c else self.t2c
        for word, color in list(t2c.items()):
            for start, end in self.find_indexes(word, self.original_text):
                self.chars[start:end].set_color(color)

    def set_color_by_t2g(self, t2g=None):
        """Internally used. Sets gradient colors for specified
        strings. Behaves similarly to ``set_color_by_t2c``."""
        t2g = t2g if t2g else self.t2g
        for word, gradient in list(t2g.items()):
            for start, end in self.find_indexes(word, self.original_text):
                self.chars[start:end].set_color_by_gradient(*gradient)

    def text2hash(self):
        """Internally used function.
        Generates ``sha256`` hash for file name.
        """
        settings = (
            "PANGO" + self.font + self.slant + self.weight
        )  # to differentiate Text and CairoText
        settings += str(self.t2f) + str(self.t2s) + str(self.t2w)
        settings += str(self.line_spacing) + str(self.size)
        settings += str(self.disable_ligatures)
        id_str = self.text + settings
        hasher = hashlib.sha256()
        hasher.update(id_str.encode())
        return hasher.hexdigest()[:16]

    def text2settings(self):
        """Internally used function. Converts the texts and styles
        to a setting for parsing."""
        settings = []
        t2x = [self.t2f, self.t2s, self.t2w]
        for i in range(len(t2x)):
            fsw = [self.font, self.slant, self.weight]
            if t2x[i]:
                for word, x in list(t2x[i].items()):
                    for start, end in self.find_indexes(word, self.text):
                        fsw[i] = x
                        settings.append(TextSetting(start, end, *fsw))
        # Set all text settings (default font, slant, weight)
        fsw = [self.font, self.slant, self.weight]
        settings.sort(key=lambda setting: setting.start)
        temp_settings = settings.copy()
        start = 0
        for setting in settings:
            if setting.start != start:
                temp_settings.append(TextSetting(start, setting.start, *fsw))
            start = setting.end
        if start != len(self.text):
            temp_settings.append(TextSetting(start, len(self.text), *fsw))
        settings = sorted(temp_settings, key=lambda setting: setting.start)

        if re.search(r"\n", self.text):
            line_num = 0
            for start, end in self.find_indexes("\n", self.text):
                for setting in settings:
                    if setting.line_num == -1:
                        setting.line_num = line_num
                    if start < setting.end:
                        line_num += 1
                        new_setting = copy.copy(setting)
                        setting.end = end
                        new_setting.start = end
                        new_setting.line_num = line_num
                        settings.append(new_setting)
                        settings.sort(key=lambda setting: setting.start)
                        break
        for setting in settings:
            if setting.line_num == -1:
                setting.line_num = 0
        return settings

    def text2svg(self):
        """Internally used function.
        Convert the text to SVG using Pango
        """
        size = self.size * 10
        line_spacing = self.line_spacing * 10
        dir_name = config.get_dir("text_dir")
        disable_liga = self.disable_ligatures
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        hash_name = self.text2hash()
        file_name = os.path.join(dir_name, hash_name) + ".svg"
        if os.path.exists(file_name):
            return file_name
        settings = self.text2settings()
        width = 600
        height = 400

        return manimpango.text2svg(
            settings,
            size,
            line_spacing,
            disable_liga,
            file_name,
            START_X,
            START_Y,
            width,
            height,
            self.text,
        )

    def init_colors(self, propagate_colors=True):
        OpenGLSVGMobject.set_style(
            self,
            fill_color=self.fill_color or self.color,
            fill_opacity=self.fill_opacity,
            stroke_color=self.stroke_color or self.color,
            stroke_width=self.stroke_width,
            stroke_opacity=self.stroke_opacity,
            recurse=propagate_colors,
        )


class OpenGLMarkupText(OpenGLSVGMobject):
    r"""Display (non-LaTeX) text rendered using `Pango <https://pango.gnome.org/>`_.

    Text objects behave like a :class:`.VGroup`-like iterable of all characters
    in the given text. In particular, slicing is possible. Text can be formatted
    using different tags:

    - ``<b>bold</b>``, ``<i>italic</i>`` and ``<b><i>bold+italic</i></b>``
    - ``<ul>underline</ul>`` and ``<s>strike through</s>``
    - ``<tt>typewriter font</tt>``
    - ``<big>bigger font</big>`` and ``<small>smaller font</small>``
    - ``<sup>superscript</sup>`` and ``<sub>subscript</sub>``
    - ``<span underline="double" underline_color="green">double underline</span>``
    - ``<span underline="error">error underline</span>``
    - ``<span overline="single" overline_color="green">overline</span>``
    - ``<span strikethrough="true" strikethrough_color="red">strikethrough</span>``
    - ``<span font_family="sans">temporary change of font</span>``
    - ``<span foreground="red">temporary change of color</span>``
    - ``<span fgcolor="red">temporary change of color</span>``
    - ``<gradient from="YELLOW" to="RED">temporary gradient</gradient>``

    For ``<span>`` markup, colors can be specified either as hex triples like ``#aabbcc`` or as named CSS colors like ``AliceBlue``.
    The ``<gradient>`` tag being handled by Manim rather than Pango, supports hex triplets or Manim constants like ``RED`` or ``RED_A``.
    If you want to use Manim constants like ``RED_A`` together with ``<span>``,
    you will need to use Python's f-String syntax as follows:
    ``f'<span foreground="{RED_A}">here you go</span>'``

    If your text contains ligatures, the :class:`MarkupText` class may incorrectly determine
    the first and last letter when creating the gradient. This is due to the fact that e.g. ``fl``
    are two characters, but might be set as one single glyph, a ligature. If your language does
    not depend on ligatures, consider setting ``disable_ligatures=True``. If you cannot or do
    not want to do without ligatures, the ``gradient`` tag supports an optional attribute ``offset``
    which can be used to compensate for that error. Usage is as follows:

    - ``<gradient from="RED" to="YELLOW" offset="1">example</gradient>`` to *start* the gradient one letter earlier
    - ``<gradient from="RED" to="YELLOW" offset=",1">example</gradient>`` to *end* the gradient one letter earlier
    - ``<gradient from="RED" to="YELLOW" offset="2,1">example</gradient>`` to *start* the gradient two letters earlier and *end* it one letter earlier

    Specifying a second offset may be necessary if the text to be colored does
    itself contain ligatures. The same can happen when using HTML entities for
    special chars.

    When using ``underline``, ``overline`` or ``strikethrough`` together with ``<gradient>`` tags, you will also need to use the offset, because
    underlines are additional paths in the final :class:`SVGMobject`, check out the corresponding example.

    Escaping of special characters: ``>`` *should* be written as ``&gt;`` whereas ``<`` and
    ``&`` *must* be written as ``&lt;`` and ``&amp;``.

    You can find more information about Pango markup formatting at the
    corresponding documentation page:
    `Pango Markup <https://developer.gnome.org/pango/1.46/pango-Markup.html>`_.
    Please be aware that not all features are supported by this class and that
    the ``<gradient>`` tag mentioned above is not supported by Pango.

    Parameters
    ----------
    text : :class:`str`
        The text that need to created as mobject.
    fill_opacity : :class:`int`
        The fill opacity with 1 meaning opaque and 0 meaning transparent.
    stroke_width : :class:`int`
        Stroke width.
    color : :class:`str`
        Global color setting for the entire text. Local overrides are possible.
    size : :class:`int`
        Font size.
    line_spacing : :class:`int`
        Line spacing.
    font : :class:`str`
        Global font setting for the entire text. Local overrides are possible.
    slant : :class:`str`
        Global slant setting, e.g. `NORMAL` or `ITALIC`. Local overrides are possible.
    weight : :class:`str`
        Global weight setting, e.g. `NORMAL` or `BOLD`. Local overrides are possible.
    gradient: :class:`tuple`
        Global gradient setting. Local overrides are possible.


    Returns
    -------
    :class:`MarkupText`
        The text displayed in form of a :class:`.VGroup`-like mobject.

    Examples
    ---------

    .. manim:: BasicMarkupExample
        :save_last_frame:

        class BasicMarkupExample(Scene):
            def construct(self):
                text1 = MarkupText("<b>foo</b> <i>bar</i> <b><i>foobar</i></b>")
                text2 = MarkupText("<s>foo</s> <u>bar</u> <big>big</big> <small>small</small>")
                text3 = MarkupText("H<sub>2</sub>O and H<sub>3</sub>O<sup>+</sup>")
                text4 = MarkupText("type <tt>help</tt> for help")
                text5 = MarkupText(
                    '<span underline="double">foo</span> <span underline="error">bar</span>'
                )
                group = VGroup(text1, text2, text3, text4, text5).arrange(DOWN)
                self.add(group)

    .. manim:: ColorExample
        :save_last_frame:

        class ColorExample(Scene):
            def construct(self):
                text1 = MarkupText(
                    f'all in red <span fgcolor="{YELLOW}">except this</span>', color=RED
                )
                text2 = MarkupText("nice gradient", gradient=(BLUE, GREEN))
                text3 = MarkupText(
                    'nice <gradient from="RED" to="YELLOW">intermediate</gradient> gradient',
                    gradient=(BLUE, GREEN),
                )
                text4 = MarkupText(
                    'fl ligature <gradient from="RED" to="YELLOW">causing trouble</gradient> here'
                )
                text5 = MarkupText(
                    'fl ligature <gradient from="RED" to="YELLOW" offset="1">defeated</gradient> with offset'
                )
                text6 = MarkupText(
                    'fl ligature <gradient from="RED" to="YELLOW" offset="1">floating</gradient> inside'
                )
                text7 = MarkupText(
                    'fl ligature <gradient from="RED" to="YELLOW" offset="1,1">floating</gradient> inside'
                )
                group = VGroup(text1, text2, text3, text4, text5, text6, text7).arrange(DOWN)
                self.add(group)

    .. manim:: UnderlineExample
        :save_last_frame:

        class UnderlineExample(Scene):
            def construct(self):
                text1 = MarkupText(
                    '<span underline="double" underline_color="green">bla</span>'
                )
                text2 = MarkupText(
                    '<span underline="single" underline_color="green">xxx</span><gradient from="#ffff00" to="RED">aabb</gradient>y'
                )
                text3 = MarkupText(
                    '<span underline="single" underline_color="green">xxx</span><gradient from="#ffff00" to="RED" offset="-1">aabb</gradient>y'
                )
                text4 = MarkupText(
                    '<span underline="double" underline_color="green">xxx</span><gradient from="#ffff00" to="RED">aabb</gradient>y'
                )
                text5 = MarkupText(
                    '<span underline="double" underline_color="green">xxx</span><gradient from="#ffff00" to="RED" offset="-2">aabb</gradient>y'
                )
                group = VGroup(text1, text2, text3, text4, text5).arrange(DOWN)
                self.add(group)

    .. manim:: FontExample
        :save_last_frame:

        class FontExample(Scene):
            def construct(self):
                text1 = MarkupText(
                    'all in sans <span font_family="serif">except this</span>', font="sans"
                )
                text2 = MarkupText(
                    '<span font_family="serif">mixing</span> <span font_family="sans">fonts</span> <span font_family="monospace">is ugly</span>'
                )
                text3 = MarkupText("special char > or &gt;")
                text4 = MarkupText("special char &lt; and &amp;")
                group = VGroup(text1, text2, text3, text4).arrange(DOWN)
                self.add(group)

    .. manim:: NewlineExample
        :save_last_frame:

        class NewlineExample(Scene):
            def construct(self):
                text = MarkupText('foooo<span foreground="red">oo\nbaa</span>aar')
                self.add(text)

    .. manim:: NoLigaturesExample
        :save_last_frame:

        class NoLigaturesExample(Scene):
            def construct(self):
                text1 = MarkupText('fl<gradient from="RED" to="GREEN">oat</gradient>ing')
                text2 = MarkupText('fl<gradient from="RED" to="GREEN">oat</gradient>ing', disable_ligatures=True)
                group = VGroup(text1, text2).arrange(DOWN)
                self.add(group)


    As :class:`MarkupText` uses Pango to render text, rendering non-English
    characters is easily possible:

    .. manim:: MultiLanguage
        :save_last_frame:

        class MultiLanguage(Scene):
            def construct(self):
                morning = MarkupText("வணக்கம்", font="sans-serif")
                japanese = MarkupText(
                    '見 角 言 谷  辛 <span fgcolor="blue">辰 辵 邑</span> 酉 釆 里!'
                )  # works as in ``Text``.
                mess = MarkupText("Multi-Language", style=BOLD)
                russ = MarkupText("Здравствуйте मस नम म ", font="sans-serif")
                hin = MarkupText("नमस्ते", font="sans-serif")
                chinese = MarkupText("臂猿「黛比」帶著孩子", font="sans-serif")
                group = VGroup(morning, japanese, mess, russ, hin, chinese).arrange(DOWN)
                self.add(group)


    Tests
    -----

    Check that the creation of :class:`~.MarkupText` works::

        >>> MarkupText('The horse does not eat cucumber salad.')
        MarkupText('The horse does not eat cucumber salad.')

    """

    def __init__(
        self,
        text: str,
        fill_opacity: int = 1,
        stroke_width: int = 0,
        color: str = WHITE,
        size: int = 1,
        line_spacing: int = -1,
        font: str = "",
        slant: str = NORMAL,
        weight: str = NORMAL,
        gradient: tuple = None,
        tab_width: int = 4,
        height: int = None,
        width: int = None,
        should_center: bool = True,
        unpack_groups: bool = True,
        disable_ligatures: bool = False,
        **kwargs,
    ):
        self.text = text
        self.color = color
        self.size = size
        self.line_spacing = line_spacing
        self.font = font
        self.slant = slant
        self.weight = weight
        self.gradient = gradient
        self.tab_width = tab_width

        self.original_text = text
        self.disable_ligatures = disable_ligatures
        text_without_tabs = text
        if "\t" in text:
            text_without_tabs = text.replace("\t", " " * self.tab_width)

        colormap = self.extract_color_tags()
        if len(colormap) > 0:
            logger.warning(
                'Using <color> tags in MarkupText is deprecated. Please use <span foreground="..."> instead.',
            )
        gradientmap = self.extract_gradient_tags()

        if not MarkupUtils.validate(self.text):
            raise ValueError(
                f"Pango cannot parse your markup in {self.text}. "
                "Please check for typos, unmatched tags or unescaped "
                "special chars like < and &.",
            )

        if self.line_spacing == -1:
            self.line_spacing = self.size + self.size * 0.3
        else:
            self.line_spacing = self.size + self.size * self.line_spacing

        file_name = self.text2svg()
        PangoUtils.remove_last_M(file_name)
        super().__init__(
            file_name,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            height=height,
            width=width,
            should_center=should_center,
            unpack_groups=unpack_groups,
            **kwargs,
        )
        self.chars = OpenGLVGroup(*self.submobjects)
        self.text = text_without_tabs.replace(" ", "").replace("\n", "")

        if config["renderer"] == "opengl":
            nppc = self.n_points_per_curve
        else:
            nppc = self.n_points_per_cubic_curve
        for each in self:
            if config["renderer"] == "opengl":
                points = each.data["points"]
            else:
                points = each.points
            if len(points) == 0:
                continue
            last = points[0]
            each.clear_points()
            for index, point in enumerate(points):
                each.append_points([point])
                if (
                    index != len(points) - 1
                    and (index + 1) % nppc == 0
                    and any(point != points[index + 1])
                ):
                    each.add_line_to(last)
                    last = points[index + 1]
            each.add_line_to(last)

        if self.gradient:
            self.set_color_by_gradient(*self.gradient)
        for col in colormap:
            self.chars[
                col["start"]
                - col["start_offset"] : col["end"]
                - col["start_offset"]
                - col["end_offset"]
            ].set_color(self._parse_color(col["color"]))
        for grad in gradientmap:
            self.chars[
                grad["start"]
                - grad["start_offset"] : grad["end"]
                - grad["start_offset"]
                - grad["end_offset"]
            ].set_color_by_gradient(
                *(self._parse_color(grad["from"]), self._parse_color(grad["to"]))
            )
        # anti-aliasing
        if height is None and width is None:
            self.scale(TEXT_MOB_SCALE_FACTOR)

    def text2hash(self):
        """Generates ``sha256`` hash for file name."""
        settings = (
            "MARKUPPANGO" + self.font + self.slant + self.weight + self.color
        )  # to differentiate from classical Pango Text
        settings += str(self.line_spacing) + str(self.size)
        settings += str(self.disable_ligatures)
        id_str = self.text + settings
        hasher = hashlib.sha256()
        hasher.update(id_str.encode())
        return hasher.hexdigest()[:16]

    def text2svg(self):
        """Convert the text to SVG using Pango."""
        size = self.size * 10
        line_spacing = self.line_spacing * 10
        dir_name = config.get_dir("text_dir")
        disable_liga = self.disable_ligatures
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        hash_name = self.text2hash()
        file_name = os.path.join(dir_name, hash_name) + ".svg"
        if os.path.exists(file_name):
            return file_name

        logger.debug(f"Setting Text {self.text}")
        return MarkupUtils.text2svg(
            f'<span foreground="{self.color}">{self.text}</span>',
            self.font,
            self.slant,
            self.weight,
            size,
            line_spacing,
            disable_liga,
            file_name,
            START_X,
            START_Y,
            600,  # width
            400,  # height
        )

    def _count_real_chars(self, s):
        """Counts characters that will be displayed.

        This is needed for partial coloring or gradients, because space
        counts to the text's `len`, but has no corresponding character."""
        count = 0
        level = 0
        # temporarily replace HTML entities by single char
        s = re.sub("&[^;]+;", "x", s)
        for c in s:
            if c == "<":
                level += 1
            if c == ">" and level > 0:
                level -= 1
            elif c != " " and c != "\t" and level == 0:
                count += 1
        return count

    def extract_gradient_tags(self):
        """Used to determine which parts (if any) of the string should be formatted
        with a gradient.

        Removes the ``<gradient>`` tag, as it is not part of Pango's markup and would cause an error.
        """
        tags = re.finditer(
            r'<gradient\s+from="([^"]+)"\s+to="([^"]+)"(\s+offset="([^"]+)")?>(.+?)</gradient>',
            self.original_text,
            re.S,
        )
        gradientmap = []
        for tag in tags:
            start = self._count_real_chars(self.original_text[: tag.start(0)])
            end = start + self._count_real_chars(tag.group(5))
            offsets = tag.group(4).split(",") if tag.group(4) else [0]
            start_offset = int(offsets[0]) if offsets[0] else 0
            end_offset = int(offsets[1]) if len(offsets) == 2 and offsets[1] else 0

            gradientmap.append(
                {
                    "start": start,
                    "end": end,
                    "from": tag.group(1),
                    "to": tag.group(2),
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                },
            )
        self.text = re.sub("<gradient[^>]+>(.+?)</gradient>", r"\1", self.text, 0, re.S)
        return gradientmap

    def _parse_color(self, col):
        """Parse color given in ``<color>`` or ``<gradient>`` tags."""
        if re.match("#[0-9a-f]{6}", col):
            return col
        else:
            return Colors[col.lower()].value

    def extract_color_tags(self):
        """Used to determine which parts (if any) of the string should be formatted
        with a custom color.

        Removes the ``<color>`` tag, as it is not part of Pango's markup and would cause an error.

        Note: Using the ``<color>`` tags is deprecated. As soon as the legacy syntax is gone, this function
        will be removed.
        """
        tags = re.finditer(
            r'<color\s+col="([^"]+)"(\s+offset="([^"]+)")?>(.+?)</color>',
            self.original_text,
            re.S,
        )

        colormap = []
        for tag in tags:
            start = self._count_real_chars(self.original_text[: tag.start(0)])
            end = start + self._count_real_chars(tag.group(4))
            offsets = tag.group(3).split(",") if tag.group(3) else [0]
            start_offset = int(offsets[0]) if offsets[0] else 0
            end_offset = int(offsets[1]) if len(offsets) == 2 and offsets[1] else 0

            colormap.append(
                {
                    "start": start,
                    "end": end,
                    "color": tag.group(1),
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                },
            )
        self.text = re.sub("<color[^>]+>(.+?)</color>", r"\1", self.text, 0, re.S)
        return colormap

    def __repr__(self):
        return f"MarkupText({repr(self.original_text)})"


@contextmanager
def register_font(font_file: typing.Union[str, Path]):
    """Temporarily add a font file to Pango's search path.

    This searches for the font_file at various places. The order it searches it described below.

    1. Absolute path.
    2. In ``assets/fonts`` folder.
    3. In ``font/`` folder.
    4. In the same directory.

    Parameters
    ----------
    font_file :
        The font file to add.

    Examples
    --------
    Use ``with register_font(...)`` to add a font file to search
    path.

    .. code-block:: python

        with register_font("path/to/font_file.ttf"):
            a = Text("Hello", font="Custom Font Name")

    Raises
    ------
    FileNotFoundError:
        If the font doesn't exists.

    AttributeError:
        If this method is used on macOS.

    Notes
    -----
    This method of adding font files also works with :class:`CairoText`.

    .. important ::

        This method is available for macOS for ``ManimPango>=v0.2.3``. Using this
        method with previous releases will raise an :class:`AttributeError` on macOS.
    """

    input_folder = Path(config.input_file).parent.resolve()
    possible_paths = [
        Path(font_file),
        input_folder / "assets/fonts" / font_file,
        input_folder / "fonts" / font_file,
        input_folder / font_file,
    ]
    for path in possible_paths:
        path = path.resolve()
        if path.exists():
            file_path = path
            logger.debug("Found file at %s", file_path.absolute())
            break
    else:
        error = f"Can't find {font_file}." f"Tried these : {possible_paths}"
        raise FileNotFoundError(error)

    try:
        assert manimpango.register_font(str(file_path))
        yield
    finally:
        manimpango.unregister_font(str(file_path))
