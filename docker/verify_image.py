"""Verification scene for the Docker image, rendered by the publish-docker workflow.

Covers LaTeX (default template, which loads most of the bundled TeX packages),
XeLaTeX with fontspec, dvisvgm, Pango text, and video encoding via PyAV::

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
        tex = MathTex(r"\int_0^1 x^2\,dx = \frac{1}{3}", r"\quad \mathds{R}")
        xetex = Tex(r"XeLaTeX: \textbf{fontspec}", tex_template=XELATEX)
        text = Text("Pango text")
        group = VGroup(tex, xetex, text).arrange(DOWN)
        self.play(Write(group), run_time=0.5)
