from __future__ import annotations

from manim import *


class ExampleScene(Scene):
    def construct(self):
        formula = MathTex(
            r"P(X=k) = ",
            "\\binom{12}{k} ",
            r"0.5^k",
            r"(1-0.5)^{12-k}",
            substrings_to_isolate=["k"],
        ).scale(1.3)
        self.play(formula.animate.set_color_by_tex("k", ORANGE))
