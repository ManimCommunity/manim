import pytest

from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "tex_mobject"


@frames_comparison
def test_color_inheritance(scene):
    """Test that Text and MarkupText correctly inherit colour from
    their parent class."""

    VMobject.set_default(color=RED)
    tex = Tex("test color inheritance")
    mathtex = MathTex("test color inheritance")
    vgr = VGroup(tex, mathtex).arrange()
    VMobject.set_default()

    scene.add(vgr)
