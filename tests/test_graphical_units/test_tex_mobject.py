from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "tex_mobject"


@frames_comparison
def test_color_inheritance(scene):
    """Test that Text and MarkupText correctly inherit colour from
    their parent class.
    """
    VMobject.set_default(color=RED)
    tex = Tex("test color inheritance")
    mathtex = MathTex("test color inheritance")
    vgr = VGroup(tex, mathtex).arrange()
    VMobject.set_default()

    scene.add(vgr)


@frames_comparison
def test_set_opacity_by_tex(scene):
    """Test that set_opacity_by_tex works correctly."""
    tex = MathTex("f(x) = y", substrings_to_isolate=["f(x)"])
    tex.set_opacity_by_tex("f(x)", 0.2, 0.5)
    scene.add(tex)


def test_preserve_tex_color():
    """Test that Tex preserves original tex colors."""
    template = TexTemplate(preamble=r"\usepackage{xcolor}")
    Tex.set_default(tex_template=template)

    txt = Tex(r"\textcolor{red}{Hello} World")
    assert len(txt[0].submobjects) == 10
    assert all(char.fill_color.to_hex() == "#FF0000" for char in txt[0][:5])  # "Hello"
    assert all(
        char.fill_color.to_hex() == WHITE.to_hex() for char in txt[0][-5:]
    )  # "World"

    txt = Tex(r"\textcolor{red}{Hello} World", color=BLUE)
    assert len(txt[0].submobjects) == 10
    assert all(char.fill_color.to_hex() == "#FF0000" for char in txt[0][:5])  # "Hello"
    assert all(
        char.fill_color.to_hex() == BLUE.to_hex() for char in txt[0][-5:]
    )  # "World"

    Tex.set_default(color=GREEN)
    txt = Tex(r"\textcolor{red}{Hello} World")
    assert len(txt[0].submobjects) == 10
    assert all(char.fill_color.to_hex() == "#FF0000" for char in txt[0][:5])  # "Hello"
    assert all(
        char.fill_color.to_hex() == GREEN.to_hex() for char in txt[0][-5:]
    )  # "World"

    Tex.set_default()
