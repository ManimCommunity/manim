from __future__ import annotations

from pathlib import Path

import pytest

from manim import MathTex, SingleStringMathTex, Tex, config


def test_MathTex(using_opengl_renderer):
    MathTex("a^2 + b^2 = c^2")
    assert Path(config.media_dir, "Tex", "eb38bdba08f46c80.svg").exists()


def test_SingleStringMathTex(using_opengl_renderer):
    SingleStringMathTex("test")
    assert Path(config.media_dir, "Tex", "5b2faa68ebf42d1e.svg").exists()


@pytest.mark.parametrize(  # : PT006
    "text_input,length_sub",
    [("{{ a }} + {{ b }} = {{ c }}", 5), (r"\frac{1}{a+b\sqrt{2}}", 1)],
)
def test_double_braces_testing(using_opengl_renderer, text_input, length_sub):
    t1 = MathTex(text_input)
    assert len(t1.submobjects) == length_sub


def test_tex(using_opengl_renderer):
    Tex("The horse does not eat cucumber salad.")
    assert Path(config.media_dir, "Tex", "f2e45e6e82d750e6.svg").exists()


def test_tex_whitespace_arg(using_opengl_renderer):
    """Check that correct number of submobjects are created per string with whitespace separator"""
    separator = "\t"
    str_part_1 = "Hello"
    str_part_2 = "world"
    str_part_3 = "It is"
    str_part_4 = "me!"
    tex = Tex(str_part_1, str_part_2, str_part_3, str_part_4, arg_separator=separator)
    assert len(tex) == 4
    assert len(tex[0]) == len("".join((str_part_1 + separator).split()))
    assert len(tex[1]) == len("".join((str_part_2 + separator).split()))
    assert len(tex[2]) == len("".join((str_part_3 + separator).split()))
    assert len(tex[3]) == len("".join(str_part_4.split()))


def test_tex_non_whitespace_arg(using_opengl_renderer):
    """Check that correct number of submobjects are created per string with non_whitespace characters"""
    separator = ","
    str_part_1 = "Hello"
    str_part_2 = "world"
    str_part_3 = "It is"
    str_part_4 = "me!"
    tex = Tex(str_part_1, str_part_2, str_part_3, str_part_4, arg_separator=separator)
    assert len(tex) == 4
    assert len(tex[0]) == len("".join((str_part_1 + separator).split()))
    assert len(tex[1]) == len("".join((str_part_2 + separator).split()))
    assert len(tex[2]) == len("".join((str_part_3 + separator).split()))
    assert len(tex[3]) == len("".join(str_part_4.split()))


def test_tex_white_space_and_non_whitespace_args(using_opengl_renderer):
    """Check that correct number of submobjects are created per string when mixing characters with whitespace"""
    separator = ", \n . \t\t"
    str_part_1 = "Hello"
    str_part_2 = "world"
    str_part_3 = "It is"
    str_part_4 = "me!"
    tex = Tex(str_part_1, str_part_2, str_part_3, str_part_4, arg_separator=separator)
    assert len(tex) == 4
    assert len(tex[0]) == len("".join((str_part_1 + separator).split()))
    assert len(tex[1]) == len("".join((str_part_2 + separator).split()))
    assert len(tex[2]) == len("".join((str_part_3 + separator).split()))
    assert len(tex[3]) == len("".join(str_part_4.split()))


def test_tex_size(using_opengl_renderer):
    """Check that the size of a :class:`Tex` string is not changed."""
    text = Tex("what").center()
    vertical = text.get_top() - text.get_bottom()
    horizontal = text.get_right() - text.get_left()
    assert round(vertical[1], 4) == 0.3512
    assert round(horizontal[0], 4) == 1.0420


def test_font_size(using_opengl_renderer):
    """Test that tex_mobject classes return
    the correct font_size value after being scaled."""
    string = MathTex(0).scale(0.3)

    assert round(string.font_size, 5) == 14.4


def test_font_size_vs_scale(using_opengl_renderer):
    """Test that scale produces the same results as .scale()"""
    num = MathTex(0, font_size=12)
    num_scale = MathTex(0).scale(1 / 4)

    assert num.height == num_scale.height


def test_changing_font_size(using_opengl_renderer):
    """Test that the font_size property properly scales tex_mobject.py classes."""
    num = Tex("0", font_size=12)
    num.font_size = 48

    assert num.height == Tex("0", font_size=48).height
