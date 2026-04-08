"""Mobjects representing text rendered using Typst.

.. important::

   The ``typst`` Python package must be installed to use these classes.
   Install it via ``pip install typst>=0.14`` or add the ``typst`` optional
   dependency group (``pip install manim[typst]``).

"""

from __future__ import annotations

__all__ = [
    "Typst",
    "TypstMath",
]

import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from manim import config
from manim.constants import DEFAULT_FONT_SIZE, SCALE_FACTOR_PER_FONT_POINT, RendererType
from manim.mobject.svg.svg_mobject import SVGMobject
from manim.mobject.types.vectorized_mobject import VGroup, VMobject
from manim.utils.color import BLACK, ParsableManimColor
from manim.utils.typst_file_writing import typst_to_svg_file

_MANIMGRP_PREAMBLE = '#let manimgrp(lbl, body) = [#box(body) #label(lbl)]'

# Pattern for the label part of {{ content : label }}.
# The label must be a valid Typst label identifier.
_LABEL_RE = re.compile(r'^(.*)\s*:\s*([a-zA-Z_][a-zA-Z0-9_-]*)\s*$', re.DOTALL)


class Typst(SVGMobject):
    """A mobject rendered from a Typst markup string.

    The Typst source is compiled to SVG via the ``typst`` Python package
    (a self-contained Rust binary extension — no system-level install
    required) and then imported through :class:`~.SVGMobject`.

    Parameters
    ----------
    typst_code
        Raw Typst markup to be compiled. This string is placed verbatim
        into the body of a minimal Typst document.
    font_size
        Font size in Manim font-size units (default: ``DEFAULT_FONT_SIZE``,
        i.e. 48). The actual scaling is applied *after* SVG import, matching
        the approach used by :class:`~.SingleStringMathTex`.
    typst_preamble
        Extra Typst code inserted before the body. Useful for ``#import``,
        ``#set``, or ``#show`` rules. Default: ``""``.
    color
        The color of the mobject. By default the standard VMobject color
        (white in dark mode). Overrides the Typst text fill color.
    stroke_width
        SVG stroke width override. If ``None`` (default), the stroke widths
        from Typst's SVG output are preserved.
    font_paths
        Optional list of additional font directories passed to the Typst
        compiler (e.g. for custom fonts not installed system-wide).
    should_center
        Whether to center the mobject after import (default ``True``).
    height
        Target height of the mobject. If ``None`` (default), the height is
        determined by ``font_size``.
    **kwargs
        Forwarded to :class:`~.SVGMobject`.

    Examples
    --------
    .. code-block:: python

        class TypstExample(Scene):
            def construct(self):
                formula = Typst(r"$ integral_a^b f(x) dif x $")
                self.play(Write(formula))

        class TypstTextExample(Scene):
            def construct(self):
                text = Typst(
                    r"*Hello* from _Typst!_",
                    font_size=72,
                )
                self.add(text)
    """

    def __init__(
        self,
        typst_code: str,
        *,
        font_size: float = DEFAULT_FONT_SIZE,
        typst_preamble: str = "",
        color: ParsableManimColor | None = None,
        stroke_width: float | None = None,
        font_paths: list[str | Path] | None = None,
        should_center: bool = True,
        height: float | None = None,
        **kwargs: Any,
    ):
        if color is None:
            color = VMobject().color

        self._font_size = font_size
        self.typst_code = typst_code
        self.typst_preamble = typst_preamble

        file_name = typst_to_svg_file(
            typst_code,
            preamble=typst_preamble,
            font_paths=font_paths,
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

        # Used for scaling via font_size property (mirrors SingleStringMathTex).
        self.initial_height = self.height

        if height is None:
            self.font_size = self._font_size

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.typst_code!r})"

    # -- font_size property (same approach as SingleStringMathTex) -----------

    @property
    def font_size(self) -> float:
        """The font size of the Typst mobject."""
        return self.height / self.initial_height / SCALE_FACTOR_PER_FONT_POINT

    @font_size.setter
    def font_size(self, val: float) -> None:
        if val <= 0:
            raise ValueError("font_size must be greater than 0.")
        if self.height > 0:
            self.scale(val / self.font_size)

    # -- SVG post-processing -------------------------------------------------

    def modify_xml_tree(self, element_tree: ET.ElementTree) -> ET.ElementTree:
        """Convert ``data-typst-label`` attributes to ``id`` before parsing.

        Typst's SVG renderer emits ``data-typst-label`` on ``<g>`` elements
        that carry a label (created via ``#box(body) <label>``).  The
        ``svgelements`` library propagates custom ``data-*`` attributes from
        parent groups to all children, making them unusable as unique group
        keys.  ``id`` attributes, on the other hand, are *not* inherited.

        This method walks the XML tree and promotes every
        ``data-typst-label`` to ``id`` (on ``<g>`` elements only), so that
        :meth:`~.SVGMobject.get_mobjects_from` can pick them up via its
        existing ``id``-based grouping logic.
        """
        # Let the base class inject default style wrappers first.
        element_tree = super().modify_xml_tree(element_tree)

        # Walk all elements regardless of namespace — ElementTree
        # qualifies tags with the namespace URI, so a bare ``"g"``
        # won't match ``{http://www.w3.org/2000/svg}g``.
        for element in element_tree.iter():
            label = element.get("data-typst-label")
            if label is not None:
                element.set("id", label)
                del element.attrib["data-typst-label"]

        return element_tree

    # -- sub-expression selection --------------------------------------------

    def select(self, key: str | int) -> VGroup:
        """Select a labeled sub-expression.

        Labels are created in the Typst source either manually via the
        ``manimgrp`` helper or automatically through the ``{{ }}``
        double-brace notation in :class:`TypstMath`.

        Parameters
        ----------
        key
            A label name (``str``) matching a ``data-typst-label`` in the
            SVG, or an integer index into the auto-numbered ``{{ }}``
            groups (``_grp-0``, ``_grp-1``, …).

        Returns
        -------
        VGroup
            The submobjects corresponding to the selected group.

        Raises
        ------
        KeyError
            If no group with the given label exists.
        IndexError
            If an integer index is out of range.

        Examples
        --------
        .. code-block:: python

            eq = TypstMath("{{ a + b : num }} / {{ c : den }} = {{ x }}")
            eq.select("num").set_color(RED)
            eq.select("den").set_color(BLUE)
            eq.select(2).set_color(GREEN)   # "x" (auto-numbered)
        """
        if isinstance(key, int):
            label = f"_grp-{key}"
            if label not in self.id_to_vgroup_dict:
                raise IndexError(
                    f"Group index {key} out of range. "
                    f"Available labels: {self._user_label_keys()}"
                )
            return self.id_to_vgroup_dict[label]

        if key not in self.id_to_vgroup_dict:
            raise KeyError(
                f"No group with label {key!r} found. "
                f"Available labels: {self._user_label_keys()}"
            )
        return self.id_to_vgroup_dict[key]

    def _user_label_keys(self) -> list[str]:
        """Return the label keys that were created from ``data-typst-label``
        attributes (filtering out internal Typst group IDs and auto-numbered
        groups).
        """
        return [
            k
            for k in self.id_to_vgroup_dict
            if not k.startswith(("numbered_group_", "root", "g"))
        ]

    # -- color handling ------------------------------------------------------

    def init_colors(self, propagate_colors: bool = True) -> Typst:
        """Recolor black submobjects to ``self.color``.

        Typst renders text in black (``fill="#000000"``) by default.
        This mirrors the approach of :meth:`SingleStringMathTex.init_colors`:
        any submobject whose color is black is recolored to ``self.color``,
        while explicitly colored submobjects (non-black) are preserved.
        """
        for submobject in self.submobjects:
            if submobject.color != BLACK:
                continue
            submobject.color = self.color
            if config.renderer == RendererType.OPENGL:
                submobject.init_colors()
            elif config.renderer == RendererType.CAIRO:
                submobject.init_colors(propagate_colors=propagate_colors)
        return self


class TypstMath(Typst):
    r"""Convenience wrapper: wraps the input in Typst math delimiters.

    The expression is rendered as a display-level equation
    (``$ ... $`` with surrounding spaces).

    Supports the ``{{ ... }}`` double-brace notation for grouping
    sub-expressions.  Each ``{{ content }}`` is wrapped in a labeled
    ``manimgrp`` call so that the resulting SVG contains identifiable
    groups accessible via :meth:`~.Typst.select`.

    Groups can optionally be given explicit labels:
    ``{{ content : label }}``.  Without a label, groups are
    auto-numbered (``_grp-0``, ``_grp-1``, …).

    Parameters
    ----------
    math_expression
        Typst math-mode content **without** the ``$ ... $`` delimiters.
        May contain ``{{ ... }}`` groups.
    **kwargs
        Forwarded to :class:`Typst`.

    Examples
    --------
    .. code-block:: python

        class DisplayMath(Scene):
            def construct(self):
                eq = TypstMath(r"sum_(k=0)^n k = (n(n+1)) / 2")
                self.add(eq)

        class GroupedMath(Scene):
            def construct(self):
                eq = TypstMath("{{ a + b : lhs }} = {{ c }}")
                eq.select("lhs").set_color(RED)
                eq.select(0).set_color(BLUE)   # "c" (auto-numbered)
                self.add(eq)
    """

    def __init__(self, math_expression: str, **kwargs: Any):
        processed, labels = self._preprocess_groups(math_expression)
        self._group_labels = labels

        # Inject the manimgrp helper when groups are present.
        if labels:
            preamble = kwargs.get("typst_preamble", "")
            if _MANIMGRP_PREAMBLE not in preamble:
                preamble = (
                    f"{_MANIMGRP_PREAMBLE}\n{preamble}" if preamble else _MANIMGRP_PREAMBLE
                )
            kwargs["typst_preamble"] = preamble

        super().__init__(f"$ {processed} $", **kwargs)

    # -- double-brace preprocessor -------------------------------------------

    @staticmethod
    def _preprocess_groups(math_expr: str) -> tuple[str, list[str]]:
        """Replace ``{{ ... }}`` groups with ``manimgrp(...)`` calls.

        Parameters
        ----------
        math_expr
            The raw math expression (without ``$ ... $`` delimiters).

        Returns
        -------
        tuple[str, list[str]]
            The processed expression and an ordered list of group labels.
        """
        result: list[str] = []
        labels: list[str] = []
        auto_index = 0
        i = 0
        n = len(math_expr)
        outer_in_string = False
        outer_bracket_depth = 0

        while i < n:
            ch = math_expr[i]

            # Track string literals at the outer level.
            if outer_in_string:
                result.append(ch)
                if ch == "\\" and i + 1 < n:
                    result.append(math_expr[i + 1])
                    i += 2
                    continue
                if ch == '"':
                    outer_in_string = False
                i += 1
                continue
            if ch == '"':
                outer_in_string = True
                result.append(ch)
                i += 1
                continue

            # Track [...] content blocks at the outer level.
            if ch == "[":
                outer_bracket_depth += 1
                result.append(ch)
                i += 1
                continue
            if ch == "]" and outer_bracket_depth > 0:
                outer_bracket_depth -= 1
                result.append(ch)
                i += 1
                continue
            if outer_bracket_depth > 0:
                result.append(ch)
                i += 1
                continue

            # Look for opening {{ (not a single {)
            if i + 1 < n and ch == "{" and math_expr[i + 1] == "{":
                i += 2  # skip {{
                content_start = i
                depth = 1
                in_string = False
                bracket_depth = 0

                while i < n and depth > 0:
                    ch = math_expr[i]

                    if in_string:
                        if ch == "\\" and i + 1 < n:
                            i += 2
                            continue
                        if ch == '"':
                            in_string = False
                        i += 1
                        continue

                    if ch == '"':
                        in_string = True
                        i += 1
                        continue

                    if ch == "[":
                        bracket_depth += 1
                        i += 1
                        continue
                    if ch == "]" and bracket_depth > 0:
                        bracket_depth -= 1
                        i += 1
                        continue
                    if bracket_depth > 0:
                        i += 1
                        continue

                    if (
                        ch == "{"
                        and i + 1 < n
                        and math_expr[i + 1] == "{"
                    ):
                        depth += 1
                        i += 2
                        continue
                    if (
                        ch == "}"
                        and i + 1 < n
                        and math_expr[i + 1] == "}"
                    ):
                        depth -= 1
                        if depth == 0:
                            content = math_expr[content_start:i]
                            i += 2  # skip }}
                            break
                        i += 2
                        continue

                    i += 1
                else:
                    # Unclosed {{ — emit literally and stop.
                    result.append("{{")
                    result.append(math_expr[content_start:])
                    return "".join(result), labels

                # Check for optional `: label` suffix.
                m = _LABEL_RE.match(content)
                if m is not None:
                    body = m.group(1).strip()
                    label = m.group(2)
                else:
                    body = content.strip()
                    label = f"_grp-{auto_index}"
                    auto_index += 1

                labels.append(label)
                result.append(f'manimgrp("{label}", {body})')
            else:
                result.append(math_expr[i])
                i += 1

        return "".join(result), labels
