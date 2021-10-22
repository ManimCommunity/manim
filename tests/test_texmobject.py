from pathlib import Path

import pytest

from manim import MathTex, SingleStringMathTex, Tex, TexTemplate, config


def test_MathTex():
    MathTex("a^2 + b^2 = c^2")
    assert Path(config.media_dir, "Tex", "3879f6b03bc495cd.svg").exists()


def test_SingleStringMathTex():
    SingleStringMathTex("test")
    assert Path(config.media_dir, "Tex", "79822967f1fa1935.svg").exists()


@pytest.mark.parametrize(  # : PT006
    "text_input,length_sub",
    [("{{ a }} + {{ b }} = {{ c }}", 5), (r"\frac{1}{a+b\sqrt{2}}", 1)],
)
def test_double_braces_testing(text_input, length_sub):
    t1 = MathTex(text_input)
    assert len(t1.submobjects) == length_sub


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


def test_tex_size():
    """Check that the size of a :class:`Tex` string is not changed."""
    text = Tex("what").center()
    vertical = text.get_top() - text.get_bottom()
    horizontal = text.get_right() - text.get_left()
    assert round(vertical[1], 4) == 0.3512
    assert round(horizontal[0], 4) == 1.0420


def test_font_size():
    """Test that tex_mobject classes return
    the correct font_size value after being scaled."""
    string = MathTex(0).scale(0.3)

    assert round(string.font_size, 5) == 14.4


def test_font_size_vs_scale():
    """Test that scale produces the same results as .scale()"""
    num = MathTex(0, font_size=12)
    num_scale = MathTex(0).scale(1 / 4)

    assert num.height == num_scale.height


def test_changing_font_size():
    """Test that the font_size property properly scales tex_mobject.py classes."""
    num = Tex("0", font_size=12)
    num.font_size = 48

    assert num.height == Tex("0", font_size=48).height


def test_log_error_context(capsys):
    """Test that the environment context of an error is correctly logged if it exists"""
    invalid_tex = r"""
        some text that is fine

        \begin{unbalanced_braces}{
        not fine
        \end{not_even_the_right_env}
        """

    with pytest.raises(ValueError) as err:
        Tex(invalid_tex)

    # validate useful TeX error logged to user
    assert "unbalanced_braces" in str(capsys.readouterr().out)
    # validate useful error message raised
    assert "See log output above or the log file" in str(err.value)


def test_log_error_no_relevant_context(capsys):
    """Test that an error with no environment context contains no environment context"""
    failing_preamble = r"""\usepackage{fontspec}
    \setmainfont[Ligatures=TeX]{not_a_font}"""

    with pytest.raises(ValueError) as err:
        Tex(
            "The template uses a non-existent font",
            tex_template=TexTemplate(preamble=failing_preamble),
        )

    # validate useless TeX error not logged for user
    assert "Context" not in str(capsys.readouterr().out)
    # validate useful error message raised
    # this won't happen if an error is raised while formatting the message
    assert "See log output above or the log file" in str(err.value)


def test_error_in_nested_context(capsys):
    """Test that displayed error context is not excessively large"""
    invalid_tex = r"""
    \begin{align}
      \begin{tabular}{ c }
        no need to display \\
        this correct text \\
      \end{tabular}
      \notacontrolsequence
    \end{align}
    """

    with pytest.raises(ValueError) as err:
        Tex(invalid_tex)

    stdout = str(capsys.readouterr().out)
    # validate useless context is not included
    assert r"\begin{frame}" not in stdout
