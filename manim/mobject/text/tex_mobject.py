r"""Mobjects representing text rendered using LaTeX.

.. important::

   See the corresponding tutorial :ref:`rendering-with-latex`

.. note::

   Just as you can use :class:`~.Text` (from the module :mod:`~.text_mobject`) to add text to your videos, you can use :class:`~.Tex` and :class:`~.MathTex` to insert LaTeX.

"""

from __future__ import annotations

from manim.utils.color import ManimColor

__all__ = [
    "SingleStringMathTex",
    "MathTex",
    "Tex",
    "BulletedList",
    "Title",
]


import itertools as it
import operator as op
import re
from functools import reduce
from textwrap import dedent
from typing import Iterable

from manim import config, logger
from manim.constants import *
from manim.mobject.geometry.line import Line
from manim.mobject.svg.svg_mobject import SVGMobject
from manim.mobject.types.vectorized_mobject import VectorizedPoint, VGroup, VMobject
from manim.utils.tex import TexTemplate
from manim.utils.tex_file_writing import tex_to_svg_file

tex_string_to_mob_map = {}


class SingleStringMathTex(SVGMobject):
    """Elementary building block for rendering text with LaTeX.

    Tests
    -----
    Check that creating a :class:`~.SingleStringMathTex` object works::

        >>> SingleStringMathTex('Test') # doctest: +SKIP
        SingleStringMathTex('Test')
    """

    def __init__(
        self,
        tex_string: str,
        stroke_width: float = 0,
        should_center: bool = True,
        height: float | None = None,
        organize_left_to_right: bool = False,
        tex_environment: str = "align*",
        tex_template: TexTemplate | None = None,
        font_size: float = DEFAULT_FONT_SIZE,
        **kwargs,
    ):
        if kwargs.get("color") is None:
            # makes it so that color isn't explicitly passed for these mobs,
            # and can instead inherit from the parent
            kwargs["color"] = VMobject().color

        self._font_size = font_size
        self.organize_left_to_right = organize_left_to_right
        self.tex_environment = tex_environment
        if tex_template is None:
            tex_template = config["tex_template"]
        self.tex_template = tex_template

        assert isinstance(tex_string, str)
        self.tex_string = tex_string
        file_name = tex_to_svg_file(
            self._get_modified_expression(tex_string),
            environment=self.tex_environment,
            tex_template=self.tex_template,
        )
        super().__init__(
            file_name=file_name,
            should_center=should_center,
            stroke_width=stroke_width,
            height=height,
            path_string_config={
                "should_subdivide_sharp_curves": True,
                "should_remove_null_curves": True,
            },
            **kwargs,
        )
        self.init_colors()

        # used for scaling via font_size.setter
        self.initial_height = self.height

        if height is None:
            self.font_size = self._font_size

        if self.organize_left_to_right:
            self._organize_submobjects_left_to_right()

    def __repr__(self):
        return f"{type(self).__name__}({repr(self.tex_string)})"

    @property
    def font_size(self):
        """The font size of the tex mobject."""
        return self.height / self.initial_height / SCALE_FACTOR_PER_FONT_POINT

    @font_size.setter
    def font_size(self, font_val):
        if font_val <= 0:
            raise ValueError("font_size must be greater than 0.")
        elif self.height > 0:
            # sometimes manim generates a SingleStringMathex mobject with 0 height.
            # can't be scaled regardless and will error without the elif.

            # scale to a factor of the initial height so that setting
            # font_size does not depend on current size.
            self.scale(font_val / self.font_size)

    def _get_modified_expression(self, tex_string):
        result = tex_string
        result = result.strip()
        result = self._modify_special_strings(result)
        return result

    def _modify_special_strings(self, tex):
        tex = tex.strip()
        should_add_filler = reduce(
            op.or_,
            [
                # Fraction line needs something to be over
                tex == "\\over",
                tex == "\\overline",
                # Make sure sqrt has overbar
                tex == "\\sqrt",
                tex == "\\sqrt{",
                # Need to add blank subscript or superscript
                tex.endswith("_"),
                tex.endswith("^"),
                tex.endswith("dot"),
            ],
        )

        if should_add_filler:
            filler = "{\\quad}"
            tex += filler

        if tex == "\\substack":
            tex = "\\quad"

        if tex == "":
            tex = "\\quad"

        # To keep files from starting with a line break
        if tex.startswith("\\\\"):
            tex = tex.replace("\\\\", "\\quad\\\\")

        # Handle imbalanced \left and \right
        num_lefts, num_rights = (
            len([s for s in tex.split(substr)[1:] if s and s[0] in "(){}[]|.\\"])
            for substr in ("\\left", "\\right")
        )
        if num_lefts != num_rights:
            tex = tex.replace("\\left", "\\big")
            tex = tex.replace("\\right", "\\big")

        tex = self._remove_stray_braces(tex)

        for context in ["array"]:
            begin_in = ("\\begin{%s}" % context) in tex
            end_in = ("\\end{%s}" % context) in tex
            if begin_in ^ end_in:
                # Just turn this into a blank string,
                # which means caller should leave a
                # stray \\begin{...} with other symbols
                tex = ""
        return tex

    def _remove_stray_braces(self, tex):
        r"""
        Makes :class:`~.MathTex` resilient to unmatched braces.

        This is important when the braces in the TeX code are spread over
        multiple arguments as in, e.g., ``MathTex(r"e^{i", r"\tau} = 1")``.
        """

        # "\{" does not count (it's a brace literal), but "\\{" counts (it's a new line and then brace)
        num_lefts = tex.count("{") - tex.count("\\{") + tex.count("\\\\{")
        num_rights = tex.count("}") - tex.count("\\}") + tex.count("\\\\}")
        while num_rights > num_lefts:
            tex = "{" + tex
            num_lefts += 1
        while num_lefts > num_rights:
            tex = tex + "}"
            num_rights += 1
        return tex

    def _organize_submobjects_left_to_right(self):
        self.sort(lambda p: p[0])
        return self

    def get_tex_string(self):
        return self.tex_string

    def init_colors(self, propagate_colors=True):
        if config.renderer == RendererType.OPENGL:
            super().init_colors()
        elif config.renderer == RendererType.CAIRO:
            super().init_colors(propagate_colors=propagate_colors)


class MathTex(SingleStringMathTex):
    r"""A string compiled with LaTeX in math mode.

    Examples
    --------
    .. manim:: Formula
        :save_last_frame:

        class Formula(Scene):
            def construct(self):
                t = MathTex(r"\int_a^b f'(x) dx = f(b)- f(a)")
                self.add(t)

    Tests
    -----
    Check that creating a :class:`~.MathTex` works::

        >>> MathTex('a^2 + b^2 = c^2') # doctest: +SKIP
        MathTex('a^2 + b^2 = c^2')

    Check that double brace group splitting works correctly::

        >>> t1 = MathTex('{{ a }} + {{ b }} = {{ c }}') # doctest: +SKIP
        >>> len(t1.submobjects) # doctest: +SKIP
        5
        >>> t2 = MathTex(r"\frac{1}{a+b\sqrt{2}}") # doctest: +SKIP
        >>> len(t2.submobjects) # doctest: +SKIP
        1

    """

    def __init__(
        self,
        *tex_strings,
        arg_separator: str = " ",
        substrings_to_isolate: Iterable[str] | None = None,
        tex_to_color_map: dict[str, ManimColor] = None,
        tex_environment: str = "align*",
        **kwargs,
    ):
        self.tex_template = kwargs.pop("tex_template", config["tex_template"])
        self.arg_separator = arg_separator
        self.substrings_to_isolate = (
            [] if substrings_to_isolate is None else substrings_to_isolate
        )
        self.tex_to_color_map = tex_to_color_map
        if self.tex_to_color_map is None:
            self.tex_to_color_map = {}
        self.tex_environment = tex_environment
        self.brace_notation_split_occurred = False
        self.tex_strings = self._break_up_tex_strings(tex_strings)
        try:
            super().__init__(
                self.arg_separator.join(self.tex_strings),
                tex_environment=self.tex_environment,
                tex_template=self.tex_template,
                **kwargs,
            )
            self._break_up_by_substrings()
        except ValueError as compilation_error:
            if self.brace_notation_split_occurred:
                logger.error(
                    dedent(
                        """\
                        A group of double braces, {{ ... }}, was detected in
                        your string. Manim splits TeX strings at the double
                        braces, which might have caused the current
                        compilation error. If you didn't use the double brace
                        split intentionally, add spaces between the braces to
                        avoid the automatic splitting: {{ ... }} --> { { ... } }.
                        """,
                    ),
                )
            raise compilation_error
        self.set_color_by_tex_to_color_map(self.tex_to_color_map)

        if self.organize_left_to_right:
            self._organize_submobjects_left_to_right()

    def _break_up_tex_strings(self, tex_strings):
        # Separate out anything surrounded in double braces
        pre_split_length = len(tex_strings)
        tex_strings = [re.split("{{(.*?)}}", str(t)) for t in tex_strings]
        tex_strings = sum(tex_strings, [])
        if len(tex_strings) > pre_split_length:
            self.brace_notation_split_occurred = True

        # Separate out any strings specified in the isolate
        # or tex_to_color_map lists.
        patterns = []
        patterns.extend(
            [
                f"({re.escape(ss)})"
                for ss in it.chain(
                    self.substrings_to_isolate,
                    self.tex_to_color_map.keys(),
                )
            ],
        )
        pattern = "|".join(patterns)
        if pattern:
            pieces = []
            for s in tex_strings:
                pieces.extend(re.split(pattern, s))
        else:
            pieces = tex_strings
        return [p for p in pieces if p]

    def _break_up_by_substrings(self):
        """
        Reorganize existing submobjects one layer
        deeper based on the structure of tex_strings (as a list
        of tex_strings)
        """
        new_submobjects = []
        curr_index = 0
        for tex_string in self.tex_strings:
            sub_tex_mob = SingleStringMathTex(
                tex_string,
                tex_environment=self.tex_environment,
                tex_template=self.tex_template,
            )
            num_submobs = len(sub_tex_mob.submobjects)
            new_index = (
                curr_index + num_submobs + len("".join(self.arg_separator.split()))
            )
            if num_submobs == 0:
                last_submob_index = min(curr_index, len(self.submobjects) - 1)
                sub_tex_mob.move_to(self.submobjects[last_submob_index], RIGHT)
            else:
                sub_tex_mob.submobjects = self.submobjects[curr_index:new_index]
            new_submobjects.append(sub_tex_mob)
            curr_index = new_index
        self.submobjects = new_submobjects
        return self

    def get_parts_by_tex(self, tex, substring=True, case_sensitive=True):
        def test(tex1, tex2):
            if not case_sensitive:
                tex1 = tex1.lower()
                tex2 = tex2.lower()
            if substring:
                return tex1 in tex2
            else:
                return tex1 == tex2

        return VGroup(*(m for m in self.submobjects if test(tex, m.get_tex_string())))

    def get_part_by_tex(self, tex, **kwargs):
        all_parts = self.get_parts_by_tex(tex, **kwargs)
        return all_parts[0] if all_parts else None

    def set_color_by_tex(self, tex, color, **kwargs):
        parts_to_color = self.get_parts_by_tex(tex, **kwargs)
        for part in parts_to_color:
            part.set_color(color)
        return self

    def set_opacity_by_tex(
        self, tex: str, opacity: float = 0.5, remaining_opacity: float = None, **kwargs
    ):
        """
        Sets the opacity of the tex specified. If 'remaining_opacity' is specified,
        then the remaining tex will be set to that opacity.

        Parameters
        ----------
        tex
            The tex to set the opacity of.
        opacity
            Default 0.5. The opacity to set the tex to
        remaining_opacity
            Default None. The opacity to set the remaining tex to.
            If None, then the remaining tex will not be changed
        """
        if remaining_opacity is not None:
            self.set_opacity(opacity=remaining_opacity)
        for part in self.get_parts_by_tex(tex):
            part.set_opacity(opacity)
        return self

    def set_color_by_tex_to_color_map(self, texs_to_color_map, **kwargs):
        for texs, color in list(texs_to_color_map.items()):
            try:
                # If the given key behaves like tex_strings
                texs + ""
                self.set_color_by_tex(texs, color, **kwargs)
            except TypeError:
                # If the given key is a tuple
                for tex in texs:
                    self.set_color_by_tex(tex, color, **kwargs)
        return self

    def index_of_part(self, part):
        split_self = self.split()
        if part not in split_self:
            raise ValueError("Trying to get index of part not in MathTex")
        return split_self.index(part)

    def index_of_part_by_tex(self, tex, **kwargs):
        part = self.get_part_by_tex(tex, **kwargs)
        return self.index_of_part(part)

    def sort_alphabetically(self):
        self.submobjects.sort(key=lambda m: m.get_tex_string())


class Tex(MathTex):
    r"""A string compiled with LaTeX in normal mode.

    Tests
    -----

    Check whether writing a LaTeX string works::

        >>> Tex('The horse does not eat cucumber salad.') # doctest: +SKIP
        Tex('The horse does not eat cucumber salad.')

    """

    def __init__(
        self, *tex_strings, arg_separator="", tex_environment="center", **kwargs
    ):
        super().__init__(
            *tex_strings,
            arg_separator=arg_separator,
            tex_environment=tex_environment,
            **kwargs,
        )


class BulletedList(Tex):
    """A bulleted list.

    Examples
    --------

    .. manim:: BulletedListExample
        :save_last_frame:

        class BulletedListExample(Scene):
            def construct(self):
                blist = BulletedList("Item 1", "Item 2", "Item 3", height=2, width=2)
                blist.set_color_by_tex("Item 1", RED)
                blist.set_color_by_tex("Item 2", GREEN)
                blist.set_color_by_tex("Item 3", BLUE)
                self.add(blist)
    """

    def __init__(
        self,
        *items,
        buff=MED_LARGE_BUFF,
        dot_scale_factor=2,
        tex_environment=None,
        **kwargs,
    ):
        self.buff = buff
        self.dot_scale_factor = dot_scale_factor
        self.tex_environment = tex_environment
        line_separated_items = [s + "\\\\" for s in items]
        super().__init__(
            *line_separated_items, tex_environment=tex_environment, **kwargs
        )
        for part in self:
            dot = MathTex("\\cdot").scale(self.dot_scale_factor)
            dot.next_to(part[0], LEFT, SMALL_BUFF)
            part.add_to_back(dot)
        self.arrange(DOWN, aligned_edge=LEFT, buff=self.buff)

    def fade_all_but(self, index_or_string, opacity=0.5):
        arg = index_or_string
        if isinstance(arg, str):
            part = self.get_part_by_tex(arg)
        elif isinstance(arg, int):
            part = self.submobjects[arg]
        else:
            raise TypeError(f"Expected int or string, got {arg}")
        for other_part in self.submobjects:
            if other_part is part:
                other_part.set_fill(opacity=1)
            else:
                other_part.set_fill(opacity=opacity)


class Title(Tex):
    """A mobject representing an underlined title.

    Examples
    --------
    .. manim:: TitleExample
        :save_last_frame:

        import manim

        class TitleExample(Scene):
            def construct(self):
                banner = ManimBanner()
                title = Title(f"Manim version {manim.__version__}")
                self.add(banner, title)

    """

    def __init__(
        self,
        *text_parts,
        include_underline=True,
        match_underline_width_to_text=False,
        underline_buff=MED_SMALL_BUFF,
        **kwargs,
    ):
        self.include_underline = include_underline
        self.match_underline_width_to_text = match_underline_width_to_text
        self.underline_buff = underline_buff
        super().__init__(*text_parts, **kwargs)
        self.to_edge(UP)
        if self.include_underline:
            underline_width = config["frame_width"] - 2
            underline = Line(LEFT, RIGHT)
            underline.next_to(self, DOWN, buff=self.underline_buff)
            if self.match_underline_width_to_text:
                underline.match_width(self)
            else:
                underline.width = underline_width
            self.add(underline)
            self.underline = underline
