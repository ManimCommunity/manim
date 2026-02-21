r"""Mobjects representing text rendered using LaTeX.

.. important::

   See the corresponding tutorial :ref:`rendering-with-latex`

.. note::

   Just as you can use :class:`~.Text` (from the module :mod:`~.text_mobject`) to add text to your videos, you can use :class:`~.Tex` and :class:`~.MathTex` to insert LaTeX.

"""

from __future__ import annotations

from manim.utils.color import BLACK, ParsableManimColor

__all__ = [
    "SingleStringMathTex",
    "MathTex",
    "Tex",
    "BulletedList",
    "Title",
]


import operator as op
import re
from collections.abc import Iterable
from functools import reduce
from textwrap import dedent
from typing import Any, Self

from manim import config, logger
from manim.constants import *
from manim.mobject.geometry.line import Line
from manim.mobject.svg.svg_mobject import SVGMobject
from manim.mobject.types.vectorized_mobject import VGroup, VMobject
from manim.utils.tex import TexTemplate
from manim.utils.tex_file_writing import tex_to_svg_file

from ..opengl.opengl_compatibility import ConvertToOpenGL

MATHTEX_SUBSTRING = "substring"


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
        tex_environment: str | None = "align*",
        tex_template: TexTemplate | None = None,
        font_size: float = DEFAULT_FONT_SIZE,
        color: ParsableManimColor | None = None,
        **kwargs: Any,
    ):
        if color is None:
            color = VMobject().color

        self._font_size = font_size
        self.organize_left_to_right = organize_left_to_right
        self.tex_environment = tex_environment
        if tex_template is None:
            tex_template = config["tex_template"]
        self.tex_template: TexTemplate = tex_template

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
            color=color,
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

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(self.tex_string)})"

    @property
    def font_size(self) -> float:
        """The font size of the tex mobject."""
        return self.height / self.initial_height / SCALE_FACTOR_PER_FONT_POINT

    @font_size.setter
    def font_size(self, font_val: float) -> None:
        if font_val <= 0:
            raise ValueError("font_size must be greater than 0.")
        elif self.height > 0:
            # sometimes manim generates a SingleStringMathex mobject with 0 height.
            # can't be scaled regardless and will error without the elif.

            # scale to a factor of the initial height so that setting
            # font_size does not depend on current size.
            self.scale(font_val / self.font_size)

    def _get_modified_expression(self, tex_string: str) -> str:
        result = tex_string
        result = result.strip()
        result = self._modify_special_strings(result)
        return result

    def _modify_special_strings(self, tex: str) -> str:
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
            begin_in = ("\\begin{%s}" % context) in tex  # noqa: UP031
            end_in = ("\\end{%s}" % context) in tex  # noqa: UP031
            if begin_in ^ end_in:
                # Just turn this into a blank string,
                # which means caller should leave a
                # stray \\begin{...} with other symbols
                tex = ""
        return tex

    def _remove_stray_braces(self, tex: str) -> str:
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

    def _organize_submobjects_left_to_right(self) -> Self:
        self.sort(lambda p: p[0])
        return self

    def get_tex_string(self) -> str:
        return self.tex_string

    def init_colors(self, propagate_colors: bool = True) -> Self:
        for submobject in self.submobjects:
            # needed to preserve original (non-black)
            # TeX colors of individual submobjects
            if submobject.color != BLACK:
                continue
            submobject.color = self.color
            if config.renderer == RendererType.OPENGL:
                submobject.init_colors()
            elif config.renderer == RendererType.CAIRO:
                submobject.init_colors(propagate_colors=propagate_colors)
        return self


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

    Notes
    -----
    Double-brace notation ``{{ ... }}`` can be used to split a single
    string argument into multiple submobjects without having to pass
    separate strings::

        MathTex(r"{{ a^2 }} + {{ b^2 }} = {{ c^2 }}")

    Each ``{{ ... }}`` group and every piece of text between groups
    becomes its own submobject, which is useful for
    :class:`~.TransformMatchingTex` animations.

    For ``{{`` to be recognised as a group opener it must appear either
    at the very start of the string or be immediately preceded by a
    whitespace character.  ``{{`` that follows non-whitespace — such as
    in ``\frac{{{n}}}{k}`` or ``a^{{2}}`` — is left untouched, so
    ordinary nested-brace LaTeX is not accidentally split.  To prevent
    an unintentional split, insert a space between the two braces:
    ``{{ ... }}`` → ``{ { ... } }``.

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
        *tex_strings: str,
        arg_separator: str = " ",
        substrings_to_isolate: Iterable[str] | None = None,
        tex_to_color_map: dict[str, ParsableManimColor] | None = None,
        tex_environment: str | None = "align*",
        **kwargs: Any,
    ):
        self.tex_template = kwargs.pop("tex_template", config["tex_template"])
        self.arg_separator = arg_separator
        self.substrings_to_isolate = (
            [] if substrings_to_isolate is None else list(substrings_to_isolate)
        )
        if tex_to_color_map is None:
            self.tex_to_color_map: dict[str, ParsableManimColor] = {}
        else:
            self.tex_to_color_map = tex_to_color_map
        self.substrings_to_isolate.extend(self.tex_to_color_map.keys())
        self.tex_environment = tex_environment
        self.brace_notation_split_occurred = False
        self.tex_strings = self._prepare_tex_strings(tex_strings)
        self.matched_strings_and_ids: list[tuple[str, str]] = []

        try:
            joined_string = self._join_tex_strings_with_unique_deliminters(
                self.tex_strings, self.substrings_to_isolate
            )
            super().__init__(
                joined_string,
                tex_environment=self.tex_environment,
                tex_template=self.tex_template,
                **kwargs,
            )
            # Save the original tex_string
            self.tex_string = self.arg_separator.join(self.tex_strings)
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

    def _prepare_tex_strings(self, tex_strings: Iterable[str]) -> list[str]:
        # Deal with the case where tex_strings contains integers instead
        # of strings.
        tex_strings_validated = [
            string if isinstance(string, str) else str(string) for string in tex_strings
        ]
        # Locate double curly bracers and split on them.
        tex_strings_validated_two = []
        for tex_string in tex_strings_validated:
            split = self._split_double_braces(tex_string)
            tex_strings_validated_two.extend(split)
        if len(tex_strings_validated_two) > len(tex_strings_validated):
            self.brace_notation_split_occurred = True
        return [string for string in tex_strings_validated_two if len(string) > 0]

    @staticmethod
    def _split_double_braces(tex_string: str) -> list[str]:
        """Split *tex_string* on Manim's ``{{ ... }}`` double-brace notation.

        Rules that avoid false positives on ordinary LaTeX source:

        * ``{{`` is only treated as a group opener when it appears at the very
          start of the string or is immediately preceded by a whitespace
          character.  Naturally-occurring ``{{`` in LaTeX is usually preceded
          by non-whitespace (e.g. ``\\frac{{{n}}}{k}`` or ``a^{{2}}``), so
          the whitespace guard eliminates the most common false positives
          without any brace-depth bookkeeping on the outer string.

        * Inside an open group the depth of *real* LaTeX braces is tracked.
          ``}}`` only closes the Manim group when the inner depth is zero,
          so ``{{ a^{b^{c}} }}`` is handled correctly.

        * Escape sequences are consumed as two-character units in priority
          order: ``\\\\`` first (escaped backslash), then ``\\{`` / ``\\}``
          (escaped braces).  This ensures e.g. ``\\\\}}`` is read as an
          escaped backslash followed by a real ``}}`` rather than as
          ``\\`` + ``\\}`` + lone ``}``.
        """
        segments: list[str] = []
        current = ""
        i = 0
        inside_manim = False
        inner_depth = 0

        while i < len(tex_string):
            # --- consume escape sequences as atomic units ---
            if tex_string[i] == "\\" and i + 1 < len(tex_string):
                next_ch = tex_string[i + 1]
                if next_ch == "\\" or next_ch in "{}":
                    # \\ (escaped backslash) checked before \{ / \} so that
                    # the second \ in \\ is never mistaken for an escape prefix.
                    current += tex_string[i : i + 2]
                    i += 2
                    continue

            if not inside_manim:
                # {{ opens a Manim group only at start-of-string or after whitespace.
                if (
                    tex_string[i : i + 2] == "{{"
                    and (i == 0 or tex_string[i - 1].isspace())
                ):
                    segments.append(current)
                    current = ""
                    inside_manim = True
                    inner_depth = 0
                    i += 2
                else:
                    current += tex_string[i]
                    i += 1
            else:
                if tex_string[i] == "{":
                    inner_depth += 1
                    current += tex_string[i]
                    i += 1
                elif tex_string[i] == "}" and inner_depth == 0 and tex_string[i : i + 2] == "}}":
                    # }} at inner depth 0 closes the Manim group.
                    segments.append(current)
                    current = ""
                    inside_manim = False
                    i += 2
                elif tex_string[i] == "}":
                    inner_depth -= 1
                    current += tex_string[i]
                    i += 1
                else:
                    current += tex_string[i]
                    i += 1

        segments.append(current)
        return segments

    def _join_tex_strings_with_unique_deliminters(
        self, tex_strings: list[str], substrings_to_isolate: Iterable[str]
    ) -> str:
        joined_string = ""
        ssIdx = 0
        for idx, tex_string in enumerate(tex_strings):
            string_part = rf"\special{{dvisvgm:raw <g id='unique{idx:03d}'>}}"
            self.matched_strings_and_ids.append((tex_string, f"unique{idx:03d}"))

            # Try to match with all substrings_to_isolate and apply the first match
            # then match again (on the rest of the string) and continue until no
            # characters are left in the string
            unprocessed_string = str(tex_string)
            processed_string = ""
            while len(unprocessed_string) > 0:
                first_match = self._locate_first_match(
                    substrings_to_isolate, unprocessed_string
                )

                if first_match:
                    processed, unprocessed_string = self._handle_match(
                        ssIdx, first_match
                    )
                    processed_string = processed_string + processed
                    ssIdx += 1
                else:
                    processed_string = processed_string + unprocessed_string
                    unprocessed_string = ""

            string_part += processed_string
            if idx < len(tex_strings) - 1:
                string_part += self.arg_separator
            string_part += r"\special{dvisvgm:raw </g>}"
            joined_string = joined_string + string_part
        return joined_string

    def _locate_first_match(
        self, substrings_to_isolate: Iterable[str], unprocessed_string: str
    ) -> re.Match | None:
        first_match_start = len(unprocessed_string)
        first_match_length = 0
        first_match = None
        for substring in substrings_to_isolate:
            match = re.match(f"(.*?)({re.escape(substring)})(.*)", unprocessed_string)
            if match and len(match.group(1)) < first_match_start:
                first_match = match
                first_match_start = len(match.group(1))
                first_match_length = len(match.group(2))
            elif match and len(match.group(1)) == first_match_start:
                # Break ties by looking at length of matches.
                if first_match_length < len(match.group(2)):
                    first_match = match
                    first_match_start = len(match.group(1))
                    first_match_length = len(match.group(2))
        return first_match

    def _handle_match(self, ssIdx: int, first_match: re.Match) -> tuple[str, str]:
        pre_match = first_match.group(1)
        matched_string = first_match.group(2)
        post_match = first_match.group(3)
        pre_string = (
            rf"\special{{dvisvgm:raw <g id='unique{ssIdx:03d}{MATHTEX_SUBSTRING}'>}}"
        )
        post_string = r"\special{dvisvgm:raw </g>}"
        self.matched_strings_and_ids.append(
            (matched_string, f"unique{ssIdx:03d}{MATHTEX_SUBSTRING}")
        )
        processed_string = pre_match + pre_string + matched_string + post_string
        unprocessed_string = post_match
        return processed_string, unprocessed_string

    @property
    def _substring_matches(self) -> list[tuple[str, str]]:
        """Return only the 'ss' (substring_to_isolate) matches."""
        return [
            (tex, id_)
            for tex, id_ in self.matched_strings_and_ids
            if id_.endswith(MATHTEX_SUBSTRING)
        ]

    @property
    def _main_matches(self) -> list[tuple[str, str]]:
        """Return only the main tex_string matches."""
        return [
            (tex, id_)
            for tex, id_ in self.matched_strings_and_ids
            if not id_.endswith(MATHTEX_SUBSTRING)
        ]

    def _break_up_by_substrings(self) -> Self:
        """
        Reorganize existing submobjects one layer
        deeper based on the structure of tex_strings (as a list
        of tex_strings)
        """
        new_submobjects: list[VMobject] = []
        try:
            for tex_string, tex_string_id in self._main_matches:
                mtp = MathTexPart()
                mtp.tex_string = tex_string
                mtp.add(*self.id_to_vgroup_dict[tex_string_id].submobjects)
                new_submobjects.append(mtp)
        except KeyError:
            logger.error(
                f"MathTex: Could not find SVG group for tex part '{tex_string}' (id: {tex_string_id}). Using fallback to root group."
            )
            new_submobjects.append(self.id_to_vgroup_dict["root"])
        self.submobjects = new_submobjects
        return self

    def get_part_by_tex(self, tex: str, **kwargs: Any) -> VGroup | None:
        for tex_str, match_id in self.matched_strings_and_ids:
            if tex_str == tex:
                return self.id_to_vgroup_dict[match_id]
        return None

    def set_color_by_tex(
        self, tex: str, color: ParsableManimColor, **kwargs: Any
    ) -> Self:
        for tex_str, match_id in self.matched_strings_and_ids:
            if tex_str == tex:
                self.id_to_vgroup_dict[match_id].set_color(color)
        return self

    def set_opacity_by_tex(
        self,
        tex: str,
        opacity: float = 0.5,
        remaining_opacity: float | None = None,
        **kwargs: Any,
    ) -> Self:
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
        for tex_str, match_id in self.matched_strings_and_ids:
            if tex_str == tex:
                self.id_to_vgroup_dict[match_id].set_opacity(opacity)
        return self

    def set_color_by_tex_to_color_map(
        self, texs_to_color_map: dict[str, ParsableManimColor], **kwargs: Any
    ) -> Self:
        for texs, color in list(texs_to_color_map.items()):
            for match in self.matched_strings_and_ids:
                if match[0] == texs:
                    self.id_to_vgroup_dict[match[1]].set_color(color)
        return self

    def index_of_part(self, part: MathTex) -> int:
        split_self = self.split()
        if part not in split_self:
            raise ValueError("Trying to get index of part not in MathTex")
        return split_self.index(part)

    def sort_alphabetically(self) -> None:
        self.submobjects.sort(key=lambda m: m.get_tex_string())


class MathTexPart(VMobject, metaclass=ConvertToOpenGL):
    tex_string: str

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(self.tex_string)})"


class Tex(MathTex):
    r"""A string compiled with LaTeX in normal mode.

    The color can be set using
    the ``color`` argument. Any parts of the ``tex_string`` that are colored by the
    TeX commands ``\color`` or ``\textcolor`` will retain their original color.

    Tests
    -----

    Check whether writing a LaTeX string works::

        >>> Tex('The horse does not eat cucumber salad.') # doctest: +SKIP
        Tex('The horse does not eat cucumber salad.')

    """

    def __init__(
        self,
        *tex_strings: str,
        arg_separator: str = "",
        tex_environment: str | None = "center",
        **kwargs: Any,
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
        *items: str,
        buff: float = MED_LARGE_BUFF,
        dot_scale_factor: float = 2,
        tex_environment: str | None = None,
        **kwargs: Any,
    ):
        self.buff = buff
        self.dot_scale_factor = dot_scale_factor
        self.tex_environment = tex_environment
        line_separated_items = [s + "\\\\" for s in items]
        super().__init__(
            *line_separated_items,
            tex_environment=tex_environment,
            **kwargs,
        )
        for part in self:
            dot = MathTex("\\cdot").scale(self.dot_scale_factor)
            dot.next_to(part[0], LEFT, SMALL_BUFF)
            part.add_to_back(dot)
        self.arrange(DOWN, aligned_edge=LEFT, buff=self.buff)

    def fade_all_but(self, index_or_string: int | str, opacity: float = 0.5) -> None:
        arg = index_or_string
        if isinstance(arg, str):
            part: VGroup | VMobject | None = self.get_part_by_tex(arg)
            if part is None:
                raise Exception(
                    f"Could not locate part by provided tex string '{arg}'."
                )
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
        *text_parts: str,
        include_underline: bool = True,
        match_underline_width_to_text: bool = False,
        underline_buff: float = MED_SMALL_BUFF,
        **kwargs: Any,
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
