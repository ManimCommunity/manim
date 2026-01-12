from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from manim import MathTex, SingleStringMathTex, Tex, TexTemplate, tempconfig


def test_MathTex(config):
    MathTex("a^2 + b^2 = c^2")
    assert Path(config.media_dir, "Tex", "e4be163a00cf424f.svg").exists()


def test_SingleStringMathTex(config):
    SingleStringMathTex("test")
    assert Path(config.media_dir, "Tex", "8ce17c7f5013209f.svg").exists()


@pytest.mark.parametrize(  # : PT006
    ("text_input", "length_sub"),
    [("{{ a }} + {{ b }} = {{ c }}", 5), (r"\frac{1}{a+b\sqrt{2}}", 1)],
)
def test_double_braces_testing(text_input, length_sub):
    t1 = MathTex(text_input)
    assert len(t1.submobjects) == length_sub


def test_tex(config):
    Tex("The horse does not eat cucumber salad.")
    assert Path(config.media_dir, "Tex", "c3945e23e546c95a.svg").exists()


def test_tex_temp_directory(tmpdir, monkeypatch):
    # Adds a test for #3060
    # It's not possible to reproduce the issue normally, because we use
    # tempconfig to change media directory to temporary directory by default
    # we partially, revert that change here.
    monkeypatch.chdir(tmpdir)
    Path(tmpdir, "media").mkdir()
    with tempconfig({"media_dir": "media"}):
        Tex("The horse does not eat cucumber salad.")
        assert Path("media", "Tex").exists()
        assert Path("media", "Tex", "c3945e23e546c95a.svg").exists()


def test_percent_char_rendering(config):
    Tex(r"\%")
    assert Path(config.media_dir, "Tex", "4a583af4d19a3adf.tex").exists()


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


def test_multi_part_tex_with_empty_parts():
    """Check that if a Tex or MathTex Mobject with multiple
    string arguments is created where some of the parts render
    as empty SVGs, then the number of family members with points
    should still be the same as the snipped in one singular part.
    """
    tex_parts = ["(-1)", "^{", "0}"]
    one_part_fomula = MathTex("".join(tex_parts))
    multi_part_formula = MathTex(*tex_parts)

    for one_part_glyph, multi_part_glyph in zip(
        one_part_fomula.family_members_with_points(),
        multi_part_formula.family_members_with_points(),
        strict=False,
    ):
        np.testing.assert_allclose(one_part_glyph.points, multi_part_glyph.points)


def test_tex_size():
    """Check that the size of a :class:`Tex` string is not changed."""
    text = Tex("what").center()
    vertical = text.get_top() - text.get_bottom()
    horizontal = text.get_right() - text.get_left()
    assert round(vertical[1], 4) == 0.3512
    assert round(horizontal[0], 4) == 1.0420


def test_font_size():
    """Test that tex_mobject classes return
    the correct font_size value after being scaled.
    """
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

    with pytest.raises(ValueError):
        Tex(invalid_tex)

    stdout = str(capsys.readouterr().out)
    # validate useless context is not included
    assert r"\begin{frame}" not in stdout


def test_tempconfig_resetting_tex_template(config):
    my_template = TexTemplate()
    my_template.preamble = "Custom preamble!"
    with tempconfig({"tex_template": my_template}):
        assert config.tex_template.preamble == "Custom preamble!"

    assert config.tex_template.preamble != "Custom preamble!"


def test_tex_garbage_collection(tmpdir, monkeypatch, config):
    monkeypatch.chdir(tmpdir)
    Path(tmpdir, "media").mkdir()
    config.media_dir = "media"

    tex_without_log = Tex("Hello World!")  # d771330b76d29ffb.tex
    assert Path("media", "Tex", "d771330b76d29ffb.tex").exists()
    assert not Path("media", "Tex", "d771330b76d29ffb.log").exists()

    config.no_latex_cleanup = True

    tex_with_log = Tex("Hello World, again!")  # da27670a37b08799.tex
    assert Path("media", "Tex", "da27670a37b08799.log").exists()
