"""Verification scene for the Docker image, rendered by the publish-docker workflow.

Covers LaTeX (default template, and the 3b1b template that loads most of the
bundled TeX packages), XeLaTeX with fontspec, dvisvgm, Pango text, and video
encoding via PyAV::

    docker run --rm -v "$PWD/docker:/test:ro" manimcommunity/manim:TAG \
        manim render -ql --media_dir /tmp/media /test/verify_image.py VerifyImage
"""

from manim import *

XELATEX = TexTemplate(
    tex_compiler="xelatex",
    output_format=".xdv",
    preamble=r"""
\usepackage{fontspec}
\usepackage{amsmath}
""",
)


class VerifyImage(Scene):
    def construct(self):
        tex = MathTex(r"\int_0^1 x^2\,dx = \frac{1}{3}")
        packages = MathTex(
            r"\mathds{R} \quad \mathscr{L} \quad \dv{f}{x}",
            tex_template=TexTemplateLibrary.threeb1b,
        )
        # calligra is a METAFONT font, so this also checks on-demand font generation.
        calligra = Tex(
            r"{\calligra Calligra}", tex_template=TexTemplateLibrary.threeb1b
        )
        xetex = Tex(r"XeLaTeX: \textbf{fontspec}", tex_template=XELATEX)
        text = Text("Pango text")
        group = VGroup(tex, packages, calligra, xetex, text).arrange(DOWN)
        self.play(Write(group), run_time=0.5)
