import pytest

from manim import MathTex, SingleStringMathTex, Tex


def test_MathTex(temp_media_dir):
    MathTex("a^2 + b^2 = c^2")
    assert (temp_media_dir / "Tex" / "3879f6b03bc495cd.svg").exists()


def test_SingleStringMathTex(temp_media_dir):
    SingleStringMathTex("test")
    assert (temp_media_dir / "Tex" / "79822967f1fa1935.svg").exists()


@pytest.mark.parametrize(
    "text_input,length_sub",
    [("{{ a }} + {{ b }} = {{ c }}", 5), (r"\frac{1}{a+b\sqrt{2}}", 1)],
)
def test_double_braces_testing(temp_media_dir, text_input, length_sub):
    t1 = MathTex(text_input)
    len(t1.submobjects) == length_sub


def test_tex(temp_media_dir):
    Tex("The horse does not eat cucumber salad.")
    assert (temp_media_dir / "Tex" / "983949cac5bdd272.svg").exists()
