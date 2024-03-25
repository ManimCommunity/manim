"""Debugging utilities."""

from __future__ import annotations

__all__ = ["print_family", "index_labels"]


from manim.mobject.mobject import Mobject
from manim.mobject.text.numbers import Integer

from ..mobject.types.vectorized_mobject import VGroup
from .color import BLACK


def print_family(mobject, n_tabs=0):
    """For debugging purposes"""
    print("\t" * n_tabs, mobject, id(mobject))
    for submob in mobject.submobjects:
        print_family(submob, n_tabs + 1)


def index_labels(
    mobject: Mobject,
    label_height: float = 0.15,
    background_stroke_width=5,
    background_stroke_color=BLACK,
    **kwargs,
):
    r"""Returns a :class:`~.VGroup` of :class:`~.Integer` mobjects
    that shows the index of each submobject.

    Useful for working with parts of complicated mobjects.

    Parameters
    ----------
    mobject
        The mobject that will have its submobjects labelled.
    label_height
        The height of the labels, by default 0.15.
    background_stroke_width
        The stroke width of the outline of the labels, by default 5.
    background_stroke_color
        The stroke color of the outline of labels.
    kwargs
        Additional parameters to be passed into the :class`~.Integer`
        mobjects used to construct the labels.

    Examples
    --------
    .. manim:: IndexLabelsExample
        :save_last_frame:

        class IndexLabelsExample(Scene):
            def construct(self):
                text = MathTex(
                    "\\frac{d}{dx}f(x)g(x)=",
                    "f(x)\\frac{d}{dx}g(x)",
                    "+",
                    "g(x)\\frac{d}{dx}f(x)",
                )

                #index the fist term in the MathTex mob
                indices = index_labels(text[0])

                text[0][1].set_color(PURPLE_B)
                text[0][8:12].set_color(DARK_BLUE)

                self.add(text, indices)
    """

    labels = VGroup()
    for n, submob in enumerate(mobject):
        label = Integer(n, **kwargs)
        label.set_stroke(
            background_stroke_color, background_stroke_width, background=True
        )
        label.height = label_height
        label.move_to(submob)
        labels.add(label)
    return labels
