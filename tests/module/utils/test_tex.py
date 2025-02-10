import pytest

from manim.utils.tex import TexTemplate, _texcode_for_environment

DEFAULT_BODY = r"""\documentclass[preview]{standalone}
\usepackage[english]{babel}
\usepackage{amsmath}
\usepackage{amssymb}
\begin{document}
YourTextHere
\end{document}"""

BODY_WITH_ADDED_PREAMBLE = r"""\documentclass[preview]{standalone}
\usepackage[english]{babel}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{testpackage}
\begin{document}
YourTextHere
\end{document}"""

BODY_WITH_PREPENDED_PREAMBLE = r"""\documentclass[preview]{standalone}
\usepackage{testpackage}
\usepackage[english]{babel}
\usepackage{amsmath}
\usepackage{amssymb}
\begin{document}
YourTextHere
\end{document}"""

BODY_WITH_ADDED_DOCUMENT = r"""\documentclass[preview]{standalone}
\usepackage[english]{babel}
\usepackage{amsmath}
\usepackage{amssymb}
\begin{document}
\boldmath
YourTextHere
\end{document}"""

BODY_REPLACE = r"""\documentclass[preview]{standalone}
\usepackage[english]{babel}
\usepackage{amsmath}
\usepackage{amssymb}
\begin{document}
\sqrt{2}
\end{document}"""

BODY_REPLACE_IN_ENV = r"""\documentclass[preview]{standalone}
\usepackage[english]{babel}
\usepackage{amsmath}
\usepackage{amssymb}
\begin{document}
\begin{align}
\sqrt{2}
\end{align}
\end{document}"""


def test_tex_template_default_body():
    template = TexTemplate()
    assert template.body == DEFAULT_BODY


def test_tex_template_preamble():
    template = TexTemplate()

    template.add_to_preamble(r"\usepackage{testpackage}")
    assert template.body == BODY_WITH_ADDED_PREAMBLE


def test_tex_template_preprend_preamble():
    template = TexTemplate()

    template.add_to_preamble(r"\usepackage{testpackage}", prepend=True)
    assert template.body == BODY_WITH_PREPENDED_PREAMBLE


def test_tex_template_document():
    template = TexTemplate()

    template.add_to_document(r"\boldmath")
    assert template.body == BODY_WITH_ADDED_DOCUMENT


def test_tex_template_texcode_for_expression():
    template = TexTemplate()

    assert template.get_texcode_for_expression(r"\sqrt{2}") == BODY_REPLACE


def test_tex_template_texcode_for_expression_in_env():
    template = TexTemplate()

    assert (
        template.get_texcode_for_expression_in_env(r"\sqrt{2}", environment="align")
        == BODY_REPLACE_IN_ENV
    )


def test_tex_template_fixed_body():
    template = TexTemplate()

    # Usually set when calling `from_file`
    template.body = "dummy"

    assert template.body == "dummy"

    with pytest.warns(
        UserWarning,
        match="This TeX template was created with a fixed body, trying to add text the preamble will have no effect.",
    ):
        template.add_to_preamble("dummys")

    with pytest.warns(
        UserWarning,
        match="This TeX template was created with a fixed body, trying to add text the document will have no effect.",
    ):
        template.add_to_document("dummy")


def test_texcode_for_environment():
    """Test that the environment is correctly extracted from the input"""
    # environment without arguments
    assert _texcode_for_environment("align*") == (r"\begin{align*}", r"\end{align*}")
    assert _texcode_for_environment("{align*}") == (r"\begin{align*}", r"\end{align*}")
    assert _texcode_for_environment(r"\begin{align*}") == (
        r"\begin{align*}",
        r"\end{align*}",
    )
    # environment with arguments
    assert _texcode_for_environment("{tabular}[t]{cccl}") == (
        r"\begin{tabular}[t]{cccl}",
        r"\end{tabular}",
    )
    assert _texcode_for_environment("tabular}{cccl") == (
        r"\begin{tabular}{cccl}",
        r"\end{tabular}",
    )
    assert _texcode_for_environment(r"\begin{tabular}[t]{cccl}") == (
        r"\begin{tabular}[t]{cccl}",
        r"\end{tabular}",
    )
