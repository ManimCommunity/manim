import pytest

from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "tex_mobject"


@frames_comparison
def test_color_inheritance(scene):
    """Test that Text and MarkupText correctly inherit colour from
    their parent class when the preserve_colors argument is unset."""

    VMobject.set_default(color=RED)

    template = config.tex_template.copy()
    template.add_to_preamble(r"\usepackage{xcolor}")

    text = r"test color \textcolor{green}{inheritance}"

    tex_inherit = Tex(text, tex_template=template)
    mathtex_inherit = MathTex(text, tex_template=template)
    tex_preserve = Tex(text, tex_template=template, preserve_colors=True)
    mathtex_preserve = MathTex(text, tex_template=template, preserve_colors=True)

    vgr = VGroup(tex_inherit, mathtex_inherit, tex_preserve, mathtex_preserve).arrange(
        DOWN
    )
    VMobject.set_default()

    scene.add(vgr)


@frames_comparison
def test_set_opacity_by_tex(scene):
    """Test that set_opacity_by_tex works correctly."""
    tex = MathTex("f(x) = y", substrings_to_isolate=["f(x)"])
    tex.set_opacity_by_tex("f(x)", 0.2, 0.5)
    scene.add(tex)
