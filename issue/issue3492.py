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


class ExampleScene2(Scene):
    def construct(self):
        formula = MathTex(
            r"P(X=k) = 0.5^k (1-0.5)^{12-k}",
        ).scale(1.3)
        print(formula.id_to_vgroup_dict)
        # formula.id_to_vgroup_dict['unique002'].set_color(RED)
        # formula.set_color_by_tex("k", ORANGE)
        self.add(formula)


class ExampleScene3(Scene):
    def construct(self):
        formula = MathTex(
            r"P(X=k) =",
            r"\binom{12}{k}",
            r"0.5^{k}",
            r"(1-0.5)^{12-k}",
            substrings_to_isolate=["k"],
        ).scale(1.3)
        for k in formula.id_to_vgroup_dict:
            print(k)
        for key in formula.id_to_vgroup_dict:
            if key[-2:] == "ss":
                formula.id_to_vgroup_dict[key].set_color(GREEN)

        # formula.id_to_vgroup_dict['unique000ss'].set_color(RED)
        # formula.id_to_vgroup_dict['unique001ss'].set_color(GREEN)
        # formula.id_to_vgroup_dict['unique002ss'].set_color(BLUE)
        # formula.id_to_vgroup_dict['unique003ss'].set_color(YELLOW)
        # formula.set_color_by_tex("k", ORANGE)
        self.add(formula)
