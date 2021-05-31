from pathlib import Path

import pytest

from manim import MathTex, SingleStringMathTex, Tex, config


def test_MathTex():
    MathTex("a^2 + b^2 = c^2")
    assert Path(config.media_dir, "Tex", "3879f6b03bc495cd.svg").exists()


def test_SingleStringMathTex():
    SingleStringMathTex("test")
    assert Path(config.media_dir, "Tex", "79822967f1fa1935.svg").exists()


@pytest.mark.parametrize(
    "text_input,length_sub",
    [("{{ a }} + {{ b }} = {{ c }}", 5), (r"\frac{1}{a+b\sqrt{2}}", 1)],
)
def test_double_braces_testing(text_input, length_sub):
    t1 = MathTex(text_input)
    len(t1.submobjects) == length_sub


def test_tex():
    Tex("The horse does not eat cucumber salad.")
    assert Path(config.media_dir, "Tex", "983949cac5bdd272.svg").exists()


def test_tex_whitespace_arg():
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


def test_tex_non_whitespace_arg():
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


def test_tex_white_space_and_non_whitespace_args():
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
