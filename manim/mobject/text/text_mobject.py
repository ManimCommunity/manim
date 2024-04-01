"""Mobjects used for displaying (non-LaTeX) text.

.. note::
   Just as you can use :class:`~.Tex` and :class:`~.MathTex` (from the module :mod:`~.tex_mobject`)
   to insert LaTeX to your videos, you can use :class:`~.Text` to to add normal text.

.. important::

   See the corresponding tutorial :ref:`using-text-objects`, especially for information about fonts.


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

            t2 = Text("2. Clustering").set_color(WHITE)

            t3 = Text("3. Regression").set_color(WHITE)

            t4 = Text("4. Prediction").set_color(WHITE)

            x = VGroup(t1, t2, t3, t4).arrange(direction=DOWN, aligned_edge=LEFT).scale(0.7).next_to(ORIGIN,DR)
            x.set_opacity(0.5)
            x.submobjects[1].set_opacity(1)
            self.add(x)

"""

from __future__ import annotations

import functools

__all__ = ["Text", "Paragraph", "MarkupText", "register_font"]


import copy
import hashlib
import os
import re
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from typing import Iterable, Sequence

import manimpango
import numpy as np
from manimpango import MarkupUtils, PangoUtils, TextSetting

from manim import config, logger
from manim.constants import *
from manim.mobject.geometry.arc import Dot
from manim.mobject.svg.svg_mobject import SVGMobject
from manim.mobject.types.vectorized_mobject import VGroup, VMobject
from manim.utils.color import ManimColor, ParsableManimColor, color_gradient
from manim.utils.deprecation import deprecated

TEXT_MOB_SCALE_FACTOR = 0.05
DEFAULT_LINE_SPACING_SCALE = 0.3
TEXT2SVG_ADJUSTMENT_FACTOR = 4.8

__all__ = ["Text", "Paragraph", "MarkupText", "register_font"]


def remove_invisible_chars(mobject: SVGMobject) -> SVGMobject:
    """Function to remove unwanted invisible characters from some mobjects.

    Parameters
    ----------
    mobject
        Any SVGMobject from which we want to remove unwanted invisible characters.

    Returns
    -------
    :class:`~.SVGMobject`
        The SVGMobject without unwanted invisible characters.
    """
    # TODO: Refactor needed
    iscode = False
    if mobject.__class__.__name__ == "Text":
        mobject = mobject[:]
    elif mobject.__class__.__name__ == "Code":
        iscode = True
        code = mobject
        mobject = mobject.code
    mobject_without_dots = VGroup()
    if mobject[0].__class__ == VGroup:
        for i in range(len(mobject)):
            mobject_without_dots.add(VGroup())
            mobject_without_dots[i].add(*(k for k in mobject[i] if k.__class__ != Dot))
    else:
        mobject_without_dots.add(*(k for k in mobject if k.__class__ != Dot))
    if iscode:
        code.code = mobject_without_dots
        return code
    return mobject_without_dots


class Paragraph(VGroup):
    r"""Display a paragraph of text.

    For a given :class:`.Paragraph` ``par``, the attribute ``par.chars`` is a
    :class:`.VGroup` containing all the lines. In this context, every line is
    constructed as a :class:`.VGroup` of characters contained in the line.


    Parameters
    ----------
    line_spacing
        Represents the spacing between lines. Defaults to -1, which means auto.
    alignment
        Defines the alignment of paragraph. Defaults to None. Possible values are "left", "right" or "center".

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

    def __init__(
        self,
        *text: Sequence[str],
        line_spacing: float = -1,
        alignment: str | None = None,
        **kwargs,
    ) -> None:
        self.line_spacing = line_spacing
        self.alignment = alignment
        self.consider_spaces_as_chars = kwargs.get("disable_ligatures", False)
        super().__init__()

        lines_str = "\n".join(list(text))
        self.lines_text = Text(lines_str, line_spacing=line_spacing, **kwargs)
        lines_str_list = lines_str.split("\n")
        self.chars = self._gen_chars(lines_str_list)

        self.lines = [list(self.chars), [self.alignment] * len(self.chars)]
        self.lines_initial_positions = [line.get_center() for line in self.lines[0]]
        self.add(*self.lines[0])
        self.move_to(np.array([0, 0, 0]))
        if self.alignment:
            self._set_all_lines_alignments(self.alignment)

    def _gen_chars(self, lines_str_list: list) -> VGroup:
        """Function to convert a list of plain strings to a VGroup of VGroups of chars.

        Parameters
        ----------
        lines_str_list
            List of plain text strings.

        Returns
        -------
        :class:`~.VGroup`
            The generated 2d-VGroup of chars.
        """
        char_index_counter = 0
        chars = self.get_group_class()()
        for line_no in range(len(lines_str_list)):
            line_str = lines_str_list[line_no]
            # Count all the characters in line_str
            # Spaces may or may not count as characters
            if self.consider_spaces_as_chars:
                char_count = len(line_str)
            else:
                char_count = 0
                for char in line_str:
                    if not char.isspace():
                        char_count += 1

            chars.add(self.get_group_class()())
            chars[line_no].add(
                *self.lines_text.chars[
                    char_index_counter : char_index_counter + char_count
                ]
            )
            char_index_counter += char_count
            if self.consider_spaces_as_chars:
                # If spaces count as characters, count the extra \n character
                # which separates Paragraph's lines to avoid issues
                char_index_counter += 1
        return chars

    def _set_all_lines_alignments(self, alignment: str) -> Paragraph:
        """Function to set all line's alignment to a specific value.

        Parameters
        ----------
        alignment
            Defines the alignment of paragraph. Possible values are "left", "right", "center".
        """
        for line_no in range(len(self.lines[0])):
            self._change_alignment_for_a_line(alignment, line_no)
        return self

    def _set_line_alignment(self, alignment: str, line_no: int) -> Paragraph:
        """Function to set one line's alignment to a specific value.

        Parameters
        ----------
        alignment
            Defines the alignment of paragraph. Possible values are "left", "right", "center".
        line_no
            Defines the line number for which we want to set given alignment.
        """
        self._change_alignment_for_a_line(alignment, line_no)
        return self

    def _set_all_lines_to_initial_positions(self) -> Paragraph:
        """Set all lines to their initial positions."""
        self.lines[1] = [None] * len(self.lines[0])
        for line_no in range(len(self.lines[0])):
            self[line_no].move_to(
                self.get_center() + self.lines_initial_positions[line_no],
            )
        return self

    def _set_line_to_initial_position(self, line_no: int) -> Paragraph:
        """Function to set one line to initial positions.

        Parameters
        ----------
        line_no
            Defines the line number for which we want to set given alignment.
        """
        self.lines[1][line_no] = None
        self[line_no].move_to(self.get_center() + self.lines_initial_positions[line_no])
        return self

    def _change_alignment_for_a_line(self, alignment: str, line_no: int) -> None:
        """Function to change one line's alignment to a specific value.

        Parameters
        ----------
        alignment
            Defines the alignment of paragraph. Possible values are "left", "right", "center".
        line_no
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


class Text(SVGMobject):
    r"""Display (non-LaTeX) text rendered using `Pango <https://pango.gnome.org/>`_.

    Text objects behave like a :class:`.VGroup`-like iterable of all characters
    in the given text. In particular, slicing is possible.

    Parameters
    ----------
    text
        The text that needs to be created as a mobject.
    font
        The font family to be used to render the text. This is either a system font or
        one loaded with `register_font()`. Note that font family names may be different
        across operating systems.
    warn_missing_font
        If True (default), Manim will issue a warning if the font does not exist in the
        (case-sensitive) list of fonts returned from `manimpango.list_fonts()`.

    Returns
    -------
    :class:`Text`
        The mobject-like :class:`.VGroup`.

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
                text1 = Text("Hello world", slant=ITALIC)
                text2 = Text("Hello world", t2s={'world':ITALIC})
                text3 = Text("Hello world", weight=BOLD)
                text4 = Text("Hello world", t2w={'world':BOLD})
                text5 = Text("Hello world", t2c={'o':YELLOW}, disable_ligatures=True)
                text6 = Text(
                    "Visit us at docs.manim.community",
                    t2c={"docs.manim.community": YELLOW},
                    disable_ligatures=True,
               )
                text6.scale(1.3).shift(DOWN)
                self.add(text1, text2, text3, text4, text5 , text6)
                Group(*self.mobjects).arrange(DOWN, buff=.8).set(height=config.frame_height-LARGE_BUFF)

    .. manim:: TextMoreCustomization
            :save_last_frame:

            class TextMoreCustomization(Scene):
                def construct(self):
                    text1 = Text(
                        'Google',
                        t2c={'[:1]': '#3174f0', '[1:2]': '#e53125',
                             '[2:3]': '#fbb003', '[3:4]': '#3174f0',
                             '[4:5]': '#269a43', '[5:]': '#e53125'}, font_size=58).scale(3)
                    self.add(text1)

    As :class:`Text` uses Pango to render text, rendering non-English
    characters is easily possible:

    .. manim:: MultipleFonts
        :save_last_frame:

        class MultipleFonts(Scene):
            def construct(self):
                morning = Text("வணக்கம்", font="sans-serif")
                japanese = Text(
                    "日本へようこそ", t2c={"日本": BLUE}
                )  # works same as ``Text``.
                mess = Text("Multi-Language", weight=BOLD)
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

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def font_list() -> list[str]:
        return manimpango.list_fonts()

    def __init__(
        self,
        text: str,
        fill_opacity: float = 1.0,
        stroke_width: float = 0,
        color: ParsableManimColor | None = None,
        font_size: float = DEFAULT_FONT_SIZE,
        line_spacing: float = -1,
        font: str = "",
        slant: str = NORMAL,
        weight: str = NORMAL,
        t2c: dict[str, str] = None,
        t2f: dict[str, str] = None,
        t2g: dict[str, tuple] = None,
        t2s: dict[str, str] = None,
        t2w: dict[str, str] = None,
        gradient: tuple = None,
        tab_width: int = 4,
        warn_missing_font: bool = True,
        # Mobject
        height: float = None,
        width: float = None,
        should_center: bool = True,
        disable_ligatures: bool = False,
        use_svg_cache: bool = False,
        **kwargs,
    ) -> None:
        self.line_spacing = line_spacing
        if font and warn_missing_font:
            fonts_list = Text.font_list()
            # handle special case of sans/sans-serif
            if font.lower() == "sans-serif":
                font = "sans"
            if font not in fonts_list:
                # check if the capitalized version is in the supported fonts
                if font.capitalize() in fonts_list:
                    font = font.capitalize()
                elif font.lower() in fonts_list:
                    font = font.lower()
                elif font.title() in fonts_list:
                    font = font.title()
                else:
                    logger.warning(f"Font {font} not in {fonts_list}.")
        self.font = font
        self._font_size = float(font_size)
        # needs to be a float or else size is inflated when font_size = 24
        # (unknown cause)
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
        self.t2c = {k: ManimColor(v).to_hex() for k, v in t2c.items()}
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
            self.line_spacing = (
                self._font_size + self._font_size * DEFAULT_LINE_SPACING_SCALE
            )
        else:
            self.line_spacing = self._font_size + self._font_size * self.line_spacing

        color: ManimColor = ManimColor(color) if color else VMobject().color
        file_name = self._text2svg(color.to_hex())
        PangoUtils.remove_last_M(file_name)
        super().__init__(
            file_name,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            height=height,
            width=width,
            should_center=should_center,
            use_svg_cache=use_svg_cache,
            **kwargs,
        )
        self.text = text
        if self.disable_ligatures:
            self.submobjects = [*self._gen_chars()]
        self.chars = self.get_group_class()(*self.submobjects)
        self.text = text_without_tabs.replace(" ", "").replace("\n", "")
        nppc = self.n_points_per_curve
        for each in self:
            if len(each.points) == 0:
                continue
            points = each.points
            curve_start = points[0]
            assert len(curve_start) == self.dim, curve_start
            # Some of the glyphs in this text might not be closed,
            # so we close them by identifying when one curve ends
            # but it is not where the next curve starts.
            # It is more efficient to temporarily create a list
            # of points and add them one at a time, then turn them
            # into a numpy array at the end, rather than creating
            # new numpy arrays every time a point or fixing line
            # is added (which is O(n^2) for numpy arrays).
            closed_curve_points = []
            # OpenGL has points be part of quadratic Bezier curves;
            # Cairo uses cubic Bezier curves.
            if nppc == 3:  # RendererType.OPENGL

                def add_line_to(end):
                    nonlocal closed_curve_points
                    start = closed_curve_points[-1]
                    closed_curve_points += [
                        start,
                        (start + end) / 2,
                        end,
                    ]

            else:  # RendererType.CAIRO

                def add_line_to(end):
                    nonlocal closed_curve_points
                    start = closed_curve_points[-1]
                    closed_curve_points += [
                        start,
                        (start + start + end) / 3,
                        (start + end + end) / 3,
                        end,
                    ]

            for index, point in enumerate(points):
                closed_curve_points.append(point)
                if (
                    index != len(points) - 1
                    and (index + 1) % nppc == 0
                    and any(point != points[index + 1])
                ):
                    # Add straight line from last point on this curve to the
                    # start point on the next curve. We represent the line
                    # as a cubic bezier curve where the two control points
                    # are half-way between the start and stop point.
                    add_line_to(curve_start)
                    curve_start = points[index + 1]
            # Make sure last curve is closed
            add_line_to(curve_start)
            each.points = np.array(closed_curve_points, ndmin=2)
        # anti-aliasing
        if height is None and width is None:
            self.scale(TEXT_MOB_SCALE_FACTOR)
        self.initial_height = self.height

    def __repr__(self):
        return f"Text({repr(self.original_text)})"

    @property
    def font_size(self):
        return (
            self.height
            / self.initial_height
            / TEXT_MOB_SCALE_FACTOR
            * 2.4
            * self._font_size
            / DEFAULT_FONT_SIZE
        )

    @font_size.setter
    def font_size(self, font_val):
        # TODO: use pango's font size scaling.
        if font_val <= 0:
            raise ValueError("font_size must be greater than 0.")
        else:
            self.scale(font_val / self.font_size)

    def _gen_chars(self):
        chars = self.get_group_class()()
        submobjects_char_index = 0
        for char_index in range(len(self.text)):
            if self.text[char_index].isspace():
                space = Dot(radius=0, fill_opacity=0, stroke_opacity=0)
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

    def _find_indexes(self, word: str, text: str):
        """Finds the indexes of ``text`` in ``word``."""
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

    @deprecated(
        since="v0.14.0",
        until="v0.15.0",
        message="This was internal function, you shouldn't be using it anyway.",
    )
    def _set_color_by_t2c(self, t2c=None):
        """Sets color for specified strings."""
        t2c = t2c if t2c else self.t2c
        for word, color in list(t2c.items()):
            for start, end in self._find_indexes(word, self.text):
                self.chars[start:end].set_color(color)

    @deprecated(
        since="v0.14.0",
        until="v0.15.0",
        message="This was internal function, you shouldn't be using it anyway.",
    )
    def _set_color_by_t2g(self, t2g=None):
        """Sets gradient colors for specified
        strings. Behaves similarly to ``set_color_by_t2c``."""
        t2g = t2g if t2g else self.t2g
        for word, gradient in list(t2g.items()):
            for start, end in self._find_indexes(word, self.text):
                self.chars[start:end].set_color_by_gradient(*gradient)

    def _text2hash(self, color: ManimColor):
        """Generates ``sha256`` hash for file name."""
        settings = (
            "PANGO" + self.font + self.slant + self.weight + str(color)
        )  # to differentiate Text and CairoText
        settings += str(self.t2f) + str(self.t2s) + str(self.t2w) + str(self.t2c)
        settings += str(self.line_spacing) + str(self._font_size)
        settings += str(self.disable_ligatures)
        id_str = self.text + settings
        hasher = hashlib.sha256()
        hasher.update(id_str.encode())
        return hasher.hexdigest()[:16]

    def _merge_settings(
        self,
        left_setting: TextSetting,
        right_setting: TextSetting,
        default_args: dict[str, Iterable[str]],
    ) -> TextSetting:
        contained = right_setting.end < left_setting.end
        new_setting = copy.copy(left_setting) if contained else copy.copy(right_setting)

        new_setting.start = right_setting.end if contained else left_setting.end
        left_setting.end = right_setting.start
        if not contained:
            right_setting.end = new_setting.start

        for arg in default_args:
            left = getattr(left_setting, arg)
            right = getattr(right_setting, arg)
            default = default_args[arg]
            if left != default and getattr(right_setting, arg) != default:
                raise ValueError(
                    f"Ambiguous style for text '{self.text[right_setting.start:right_setting.end]}':"
                    + f"'{arg}' cannot be both '{left}' and '{right}'."
                )
            setattr(right_setting, arg, left if left != default else right)
        return new_setting

    def _get_settings_from_t2xs(
        self,
        t2xs: Sequence[tuple[dict[str, str], str]],
        default_args: dict[str, Iterable[str]],
    ) -> Sequence[TextSetting]:
        settings = []
        t2xwords = set(chain(*([*t2x.keys()] for t2x, _ in t2xs)))
        for word in t2xwords:
            setting_args = {
                arg: str(t2x[word]) if word in t2x else default_args[arg]
                # NOTE: when t2x[word] is a ManimColor, str will yield the
                # hex representation
                for t2x, arg in t2xs
            }

            for start, end in self._find_indexes(word, self.text):
                settings.append(TextSetting(start, end, **setting_args))
        return settings

    def _get_settings_from_gradient(
        self, default_args: dict[str, Iterable[str]]
    ) -> Sequence[TextSetting]:
        settings = []
        args = copy.copy(default_args)
        if self.gradient:
            colors = color_gradient(self.gradient, len(self.text))
            for i in range(len(self.text)):
                args["color"] = colors[i].to_hex()
                settings.append(TextSetting(i, i + 1, **args))

        for word, gradient in self.t2g.items():
            if isinstance(gradient, str) or len(gradient) == 1:
                color = gradient if isinstance(gradient, str) else gradient[0]
                gradient = [ManimColor(color)]
            colors = (
                color_gradient(gradient, len(word))
                if len(gradient) != 1
                else len(word) * gradient
            )
            for start, end in self._find_indexes(word, self.text):
                for i in range(start, end):
                    args["color"] = colors[i - start].to_hex()
                    settings.append(TextSetting(i, i + 1, **args))
        return settings

    def _text2settings(self, color: str):
        """Converts the texts and styles to a setting for parsing."""
        t2xs = [
            (self.t2f, "font"),
            (self.t2s, "slant"),
            (self.t2w, "weight"),
            (self.t2c, "color"),
        ]
        # setting_args requires values to be strings

        default_args = {
            arg: getattr(self, arg) if arg != "color" else color for _, arg in t2xs
        }

        settings = self._get_settings_from_t2xs(t2xs, default_args)
        settings.extend(self._get_settings_from_gradient(default_args))

        # Handle overlaps

        settings.sort(key=lambda setting: setting.start)
        for index, setting in enumerate(settings):
            if index + 1 == len(settings):
                break

            next_setting = settings[index + 1]
            if setting.end > next_setting.start:
                new_setting = self._merge_settings(setting, next_setting, default_args)
                new_index = index + 1
                while (
                    new_index < len(settings)
                    and settings[new_index].start < new_setting.start
                ):
                    new_index += 1
                settings.insert(new_index, new_setting)

        # Set all text settings (default font, slant, weight)
        temp_settings = settings.copy()
        start = 0
        for setting in settings:
            if setting.start != start:
                temp_settings.append(TextSetting(start, setting.start, **default_args))
            start = setting.end
        if start != len(self.text):
            temp_settings.append(TextSetting(start, len(self.text), **default_args))
        settings = sorted(temp_settings, key=lambda setting: setting.start)

        line_num = 0
        if re.search(r"\n", self.text):
            for start, end in self._find_indexes("\n", self.text):
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
                setting.line_num = line_num

        return settings

    def _text2svg(self, color: ManimColor):
        """Convert the text to SVG using Pango."""
        size = self._font_size
        line_spacing = self.line_spacing
        size /= TEXT2SVG_ADJUSTMENT_FACTOR
        line_spacing /= TEXT2SVG_ADJUSTMENT_FACTOR

        dir_name = config.get_dir("text_dir")
        if not dir_name.is_dir():
            dir_name.mkdir(parents=True)
        hash_name = self._text2hash(color)
        file_name = dir_name / (hash_name + ".svg")

        if file_name.exists():
            svg_file = str(file_name.resolve())
        else:
            settings = self._text2settings(color)
            width = config["pixel_width"]
            height = config["pixel_height"]

            svg_file = manimpango.text2svg(
                settings,
                size,
                line_spacing,
                self.disable_ligatures,
                str(file_name.resolve()),
                START_X,
                START_Y,
                width,
                height,
                self.text,
            )

        return svg_file

    def init_colors(self, propagate_colors=True):
        if config.renderer == RendererType.OPENGL:
            super().init_colors()
        elif config.renderer == RendererType.CAIRO:
            super().init_colors(propagate_colors=propagate_colors)


class MarkupText(SVGMobject):
    r"""Display (non-LaTeX) text rendered using `Pango <https://pango.gnome.org/>`_.

    Text objects behave like a :class:`.VGroup`-like iterable of all characters
    in the given text. In particular, slicing is possible.

    **What is PangoMarkup?**

    PangoMarkup is a small markup language like html and it helps you avoid using
    "range of characters" while coloring or styling a piece a Text. You can use
    this language with :class:`~.MarkupText`.

    A simple example of a marked-up string might be::

        <span foreground="blue" size="x-large">Blue text</span> is <i>cool</i>!"

    and it can be used with :class:`~.MarkupText` as

    .. manim:: MarkupExample
        :save_last_frame:

        class MarkupExample(Scene):
            def construct(self):
                text = MarkupText('<span foreground="blue" size="x-large">Blue text</span> is <i>cool</i>!"')
                self.add(text)

    A more elaborate example would be:

    .. manim:: MarkupElaborateExample
        :save_last_frame:

        class MarkupElaborateExample(Scene):
            def construct(self):
                text = MarkupText(
                    '<span foreground="purple">ا</span><span foreground="red">َ</span>'
                    'ل<span foreground="blue">ْ</span>ع<span foreground="red">َ</span>ر'
                    '<span foreground="red">َ</span>ب<span foreground="red">ِ</span>ي'
                    '<span foreground="green">ّ</span><span foreground="red">َ</span>ة'
                    '<span foreground="blue">ُ</span>'
                )
                self.add(text)

    PangoMarkup can also contain XML features such as numeric character
    entities such as ``&#169;`` for © can be used too.

    The most general markup tag is ``<span>``, then there are some
    convenience tags.

    Here is a list of supported tags:

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

    For ``<span>`` markup, colors can be specified either as
    hex triples like ``#aabbcc`` or as named CSS colors like
    ``AliceBlue``.
    The ``<gradient>`` tag is handled by Manim rather than
    Pango, and supports hex triplets or Manim constants like
    ``RED`` or ``RED_A``.
    If you want to use Manim constants like ``RED_A`` together
    with ``<span>``, you will need to use Python's f-String
    syntax as follows::

        MarkupText(f'<span foreground="{RED_A}">here you go</span>')

    If your text contains ligatures, the :class:`MarkupText` class may
    incorrectly determine the first and last letter when creating the
    gradient. This is due to the fact that ``fl`` are two separate characters,
    but might be set as one single glyph - a ligature. If your language
    does not depend on ligatures, consider setting ``disable_ligatures``
    to ``True``. If you must use ligatures, the ``gradient`` tag supports an optional
    attribute ``offset`` which can be used to compensate for that error.

    For example:

    - ``<gradient from="RED" to="YELLOW" offset="1">example</gradient>`` to *start* the gradient one letter earlier
    - ``<gradient from="RED" to="YELLOW" offset=",1">example</gradient>`` to *end* the gradient one letter earlier
    - ``<gradient from="RED" to="YELLOW" offset="2,1">example</gradient>`` to *start* the gradient two letters earlier and *end* it one letter earlier

    Specifying a second offset may be necessary if the text to be colored does
    itself contain ligatures. The same can happen when using HTML entities for
    special chars.

    When using ``underline``, ``overline`` or ``strikethrough`` together with
    ``<gradient>`` tags, you will also need to use the offset, because
    underlines are additional paths in the final :class:`SVGMobject`.
    Check out the following example.

    Escaping of special characters: ``>`` **should** be written as ``&gt;``
    whereas ``<`` and ``&`` *must* be written as ``&lt;`` and
    ``&amp;``.

    You can find more information about Pango markup formatting at the
    corresponding documentation page:
    `Pango Markup <https://docs.gtk.org/Pango/pango_markup.html>`_.
    Please be aware that not all features are supported by this class and that
    the ``<gradient>`` tag mentioned above is not supported by Pango.

    Parameters
    ----------

    text
        The text that needs to be created as mobject.
    fill_opacity
        The fill opacity, with 1 meaning opaque and 0 meaning transparent.
    stroke_width
        Stroke width.
    font_size
        Font size.
    line_spacing
        Line spacing.
    font
        Global font setting for the entire text. Local overrides are possible.
    slant
        Global slant setting, e.g. `NORMAL` or `ITALIC`. Local overrides are possible.
    weight
        Global weight setting, e.g. `NORMAL` or `BOLD`. Local overrides are possible.
    gradient
        Global gradient setting. Local overrides are possible.
    warn_missing_font
        If True (default), Manim will issue a warning if the font does not exist in the
        (case-sensitive) list of fonts returned from `manimpango.list_fonts()`.

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
                    '<span fgcolor="blue">日本</span>へようこそ'
                )  # works as in ``Text``.
                mess = MarkupText("Multi-Language", weight=BOLD)
                russ = MarkupText("Здравствуйте मस नम म ", font="sans-serif")
                hin = MarkupText("नमस्ते", font="sans-serif")
                chinese = MarkupText("臂猿「黛比」帶著孩子", font="sans-serif")
                group = VGroup(morning, japanese, mess, russ, hin, chinese).arrange(DOWN)
                self.add(group)

    You can justify the text by passing :attr:`justify` parameter.

    .. manim:: JustifyText

        class JustifyText(Scene):
            def construct(self):
                ipsum_text = (
                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
                    "Praesent feugiat metus sit amet iaculis pulvinar. Nulla posuere "
                    "quam a ex aliquam, eleifend consectetur tellus viverra. Aliquam "
                    "fermentum interdum justo, nec rutrum elit pretium ac. Nam quis "
                    "leo pulvinar, dignissim est at, venenatis nisi."
                )
                justified_text = MarkupText(ipsum_text, justify=True).scale(0.4)
                not_justified_text = MarkupText(ipsum_text, justify=False).scale(0.4)
                just_title = Title("Justified")
                njust_title = Title("Not Justified")
                self.add(njust_title, not_justified_text)
                self.play(
                    FadeOut(not_justified_text),
                    FadeIn(justified_text),
                    FadeOut(njust_title),
                    FadeIn(just_title),
                )
                self.wait(1)

    Tests
    -----

    Check that the creation of :class:`~.MarkupText` works::

        >>> MarkupText('The horse does not eat cucumber salad.')
        MarkupText('The horse does not eat cucumber salad.')

    """

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def font_list() -> list[str]:
        return manimpango.list_fonts()

    def __init__(
        self,
        text: str,
        fill_opacity: float = 1,
        stroke_width: float = 0,
        color: ParsableManimColor | None = None,
        font_size: float = DEFAULT_FONT_SIZE,
        line_spacing: int = -1,
        font: str = "",
        slant: str = NORMAL,
        weight: str = NORMAL,
        justify: bool = False,
        gradient: tuple = None,
        tab_width: int = 4,
        height: int = None,
        width: int = None,
        should_center: bool = True,
        disable_ligatures: bool = False,
        warn_missing_font: bool = True,
        **kwargs,
    ) -> None:
        self.text = text
        self.line_spacing = line_spacing
        if font and warn_missing_font:
            fonts_list = Text.font_list()
            # handle special case of sans/sans-serif
            if font.lower() == "sans-serif":
                font = "sans"
            if font not in fonts_list:
                # check if the capitalized version is in the supported fonts
                if font.capitalize() in fonts_list:
                    font = font.capitalize()
                elif font.lower() in fonts_list:
                    font = font.lower()
                elif font.title() in fonts_list:
                    font = font.title()
                else:
                    logger.warning(f"Font {font} not in {fonts_list}.")
        self.font = font
        self._font_size = float(font_size)
        self.slant = slant
        self.weight = weight
        self.gradient = gradient
        self.tab_width = tab_width
        self.justify = justify

        self.original_text = text
        self.disable_ligatures = disable_ligatures
        text_without_tabs = text
        if "\t" in text:
            text_without_tabs = text.replace("\t", " " * self.tab_width)

        colormap = self._extract_color_tags()
        if len(colormap) > 0:
            logger.warning(
                'Using <color> tags in MarkupText is deprecated. Please use <span foreground="..."> instead.',
            )
        gradientmap = self._extract_gradient_tags()
        validate_error = MarkupUtils.validate(self.text)
        if validate_error:
            raise ValueError(validate_error)

        if self.line_spacing == -1:
            self.line_spacing = (
                self._font_size + self._font_size * DEFAULT_LINE_SPACING_SCALE
            )
        else:
            self.line_spacing = self._font_size + self._font_size * self.line_spacing

        color: ManimColor = ManimColor(color) if color else VMobject().color
        file_name = self._text2svg(color)

        PangoUtils.remove_last_M(file_name)
        super().__init__(
            file_name,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            height=height,
            width=width,
            should_center=should_center,
            **kwargs,
        )

        self.chars = self.get_group_class()(*self.submobjects)
        self.text = text_without_tabs.replace(" ", "").replace("\n", "")

        nppc = self.n_points_per_curve
        for each in self:
            if len(each.points) == 0:
                continue
            points = each.points
            curve_start = points[0]
            assert len(curve_start) == self.dim, curve_start
            # Some of the glyphs in this text might not be closed,
            # so we close them by identifying when one curve ends
            # but it is not where the next curve starts.
            # It is more efficient to temporarily create a list
            # of points and add them one at a time, then turn them
            # into a numpy array at the end, rather than creating
            # new numpy arrays every time a point or fixing line
            # is added (which is O(n^2) for numpy arrays).
            closed_curve_points = []
            # OpenGL has points be part of quadratic Bezier curves;
            # Cairo uses cubic Bezier curves.
            if nppc == 3:  # RendererType.OPENGL

                def add_line_to(end):
                    nonlocal closed_curve_points
                    start = closed_curve_points[-1]
                    closed_curve_points += [
                        start,
                        (start + end) / 2,
                        end,
                    ]

            else:  # RendererType.CAIRO

                def add_line_to(end):
                    nonlocal closed_curve_points
                    start = closed_curve_points[-1]
                    closed_curve_points += [
                        start,
                        (start + start + end) / 3,
                        (start + end + end) / 3,
                        end,
                    ]

            for index, point in enumerate(points):
                closed_curve_points.append(point)
                if (
                    index != len(points) - 1
                    and (index + 1) % nppc == 0
                    and any(point != points[index + 1])
                ):
                    # Add straight line from last point on this curve to the
                    # start point on the next curve.
                    add_line_to(curve_start)
                    curve_start = points[index + 1]
            # Make sure last curve is closed
            add_line_to(curve_start)
            each.points = np.array(closed_curve_points, ndmin=2)

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

        self.initial_height = self.height

    @property
    def font_size(self):
        return (
            self.height
            / self.initial_height
            / TEXT_MOB_SCALE_FACTOR
            * 2.4
            * self._font_size
            / DEFAULT_FONT_SIZE
        )

    @font_size.setter
    def font_size(self, font_val):
        # TODO: use pango's font size scaling.
        if font_val <= 0:
            raise ValueError("font_size must be greater than 0.")
        else:
            self.scale(font_val / self.font_size)

    def _text2hash(self, color: ParsableManimColor):
        """Generates ``sha256`` hash for file name."""
        settings = (
            "MARKUPPANGO"
            + self.font
            + self.slant
            + self.weight
            + ManimColor(color).to_hex().lower()
        )  # to differentiate from classical Pango Text
        settings += str(self.line_spacing) + str(self._font_size)
        settings += str(self.disable_ligatures)
        settings += str(self.justify)
        id_str = self.text + settings
        hasher = hashlib.sha256()
        hasher.update(id_str.encode())
        return hasher.hexdigest()[:16]

    def _text2svg(self, color: ParsableManimColor | None):
        """Convert the text to SVG using Pango."""
        color = ManimColor(color)
        size = self._font_size
        line_spacing = self.line_spacing
        size /= TEXT2SVG_ADJUSTMENT_FACTOR
        line_spacing /= TEXT2SVG_ADJUSTMENT_FACTOR

        dir_name = config.get_dir("text_dir")
        if not dir_name.is_dir():
            dir_name.mkdir(parents=True)
        hash_name = self._text2hash(color)
        file_name = dir_name / (hash_name + ".svg")

        if file_name.exists():
            svg_file = str(file_name.resolve())
        else:
            final_text = (
                f'<span foreground="{color.to_hex()}">{self.text}</span>'
                if color is not None
                else self.text
            )
            logger.debug(f"Setting Text {self.text}")
            svg_file = MarkupUtils.text2svg(
                final_text,
                self.font,
                self.slant,
                self.weight,
                size,
                line_spacing,
                self.disable_ligatures,
                str(file_name.resolve()),
                START_X,
                START_Y,
                600,  # width
                400,  # height
                justify=self.justify,
                pango_width=500,
            )
        return svg_file

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

    def _extract_gradient_tags(self):
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
            return ManimColor(col).to_hex()

    def _extract_color_tags(self):
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
def register_font(font_file: str | Path):
    """Temporarily add a font file to Pango's search path.

    This searches for the font_file at various places. The order it searches it described below.

    1. Absolute path.
    2. In ``assets/fonts`` folder.
    3. In ``font/`` folder.
    4. In the same directory.

    Parameters
    ----------
    font_file
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
