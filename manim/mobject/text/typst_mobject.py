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

from pathlib import Path
from typing import Any

from manim import config, logger
from manim.constants import DEFAULT_FONT_SIZE, SCALE_FACTOR_PER_FONT_POINT, RendererType
from manim.mobject.svg.svg_mobject import SVGMobject
from manim.mobject.types.vectorized_mobject import VMobject
from manim.utils.color import BLACK, ParsableManimColor
from manim.utils.typst_file_writing import typst_to_svg_file


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
        SVG stroke width (default 0, matching :class:`~.MathTex` behavior).
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
        stroke_width: float = 0,
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

    Parameters
    ----------
    math_expression
        Typst math-mode content **without** the ``$ ... $`` delimiters.
    **kwargs
        Forwarded to :class:`Typst`.

    Examples
    --------
    .. code-block:: python

        class DisplayMath(Scene):
            def construct(self):
                eq = TypstMath(r"sum_(k=0)^n k = (n(n+1)) / 2")
                self.add(eq)
    """

    def __init__(self, math_expression: str, **kwargs: Any):
        super().__init__(f"$ {math_expression} $", **kwargs)
