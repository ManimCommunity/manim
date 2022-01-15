"""Debugging utilities."""


__all__ = ["print_family", "index_labels"]


from manim.mobject.mobject import Mobject
from ..mobject.numbers import Integer
from ..mobject.types.vectorized_mobject import VGroup
from .color import BLACK, Colors


def print_family(mobject, n_tabs=0):
    """For debugging purposes"""
    print("\t" * n_tabs, mobject, id(mobject))
    for submob in mobject.submobjects:
        print_family(submob, n_tabs + 1)


def index_labels(mobject: "Mobject", font_size:float=7.2, stroke_color: Colors = BLACK):
    """Returns a :class:`~.VGroup` of :class:`~.Integer`s
    that shows the index of each submobject.

    Useful for working with parts of complicated mobjects.

    Parameters
    ----------

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
        label = Integer(n, font_size=font_size)
        label.move_to(submob)
        label.set_stroke(stroke_color, 5, background=True)
        labels.add(label)
    return labels
