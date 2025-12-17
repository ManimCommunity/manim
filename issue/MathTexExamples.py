from __future__ import annotations

from manim import *


class ExampleScene2(Scene):
    def construct(self):
        formula = MathTex(
            r"P(X=k) = 0.5^k (1-0.5)^{12-k}",
        ).scale(1.3)
        print(formula.id_to_vgroup_dict.keys())
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


class ExampleScene4a(Scene):
    def construct(self):
        formula = MathTex(
            r"a^2 + b^2 = c^2 + a^2",
            substrings_to_isolate=["a", "b"],
        ).scale(1.3)
        for k in formula.id_to_vgroup_dict:
            print(k)
        for key in formula.id_to_vgroup_dict:
            if key[-2:] == "ss":
                formula.id_to_vgroup_dict[key].set_color(GREEN)

        self.add(formula)


class ExampleScene4b(Scene):
    def construct(self):
        formula = MathTex(
            r"a^2 + b^2 = c^2 + a^2",
            substrings_to_isolate=["c", "a"],
        ).scale(1.3)
        print("Hejsa")
        for k in formula.id_to_vgroup_dict:
            print(k)
        for key in formula.id_to_vgroup_dict:
            if key[-2:] == "ss":
                formula.id_to_vgroup_dict[key].set_color(GREEN)

        self.add(formula)


class ExampleScene5(Scene):
    def construct(self):
        formula = MathTex(
            r"a^2 + b^2 = c^2 + d^2 - a^2",
            substrings_to_isolate=["[acd]"],
        ).scale(1.3)
        for k in formula.id_to_vgroup_dict:
            print(k)
        for key in formula.id_to_vgroup_dict:
            if key[-2:] == "ss":
                formula.id_to_vgroup_dict[key].set_color(GREEN)

        self.add(formula)


# TODO:
# When all scenes are rendered with a single command line call
# uv run manim render MathTexExamples.py --write_all
# ExampleScene6 fails with the following error
# KeyError: 'unique001ss'
# I think it is related to a caching issue, because the error vanishes
# when the scene is rendered by itself.
# uv run manim render MathTexExamples.py ExampleScene6
class ExampleScene6(Scene):
    def construct(self):
        formula = MathTex(
            r"a^2 + b^2 = c^2 + d^2 - a^2",
            substrings_to_isolate=["[acd]"],
        ).scale(1.3)

        for k in formula.id_to_vgroup_dict:
            print(k)

        def set_color_by_tex(mathtex, tex, color):
            print(mathtex.matched_strings_and_ids)
            for match in mathtex.matched_strings_and_ids:
                if match[0] == tex:
                    mathtex.id_to_vgroup_dict[match[1]].set_color(color)

        set_color_by_tex(formula, "c", ORANGE)
        set_color_by_tex(formula, "a", RED)

        self.add(formula)


class ExampleScene7(Scene):
    def construct(self):
        formula = MathTex(
            r"a^2 + b^2 = c^2 + d^2 - 2 a^2",
            substrings_to_isolate=["[acd]"],
        ).scale(1.3)

        for k in formula.id_to_vgroup_dict:
            print(k)

        def set_color_by_tex(mathtex, tex, color):
            print(mathtex.matched_strings_and_ids)
            for match in mathtex.matched_strings_and_ids:
                if match[0] == tex:
                    mathtex.id_to_vgroup_dict[match[1]].set_color(color)

        set_color_by_tex(formula, "c", GREEN)
        set_color_by_tex(formula, "a", RED)

        self.add(formula)


class ExampleScene8(Scene):
    def construct(self):
        formula = MathTex(
            r"P(X=k) =",
            r"\binom{12}{k}",
            r"0.5^{k}",
            r"(1-0.5)^{12-k}",
            substrings_to_isolate=["k", "1", "12", "0.5"],
        ).scale(1.3)

        def set_color_by_tex(
            mathtex: MathTex, tex: str, color: ParsableManimColor
        ) -> None:
            for match in mathtex.matched_strings_and_ids:
                if match[0] == tex:
                    mathtex.id_to_vgroup_dict[match[1]].set_color(color)

        set_color_by_tex(formula, "k", GREEN)
        set_color_by_tex(formula, "12", RED)
        set_color_by_tex(formula, "1", YELLOW)
        set_color_by_tex(formula, "0.5", BLUE_D)
        self.add(formula)


class ExampleScene9(Scene):
    def construct(self):
        t2cm = {r"\sum": BLUE, "^{n}": RED, "_{1}": GREEN, "x": YELLOW}
        eq1 = MathTex(r"\sum", "^{n}", "_{1}", "x").scale(1.3)
        eq2 = MathTex(r"\sum", "_{1}", "^{n}", "x").scale(1.3)

        def set_color_by_tex(
            mathtex: MathTex, tex: str, color: ParsableManimColor
        ) -> None:
            for match in mathtex.matched_strings_and_ids:
                if match[0] == tex:
                    mathtex.id_to_vgroup_dict[match[1]].set_color(color)

        for k, v in t2cm.items():
            set_color_by_tex(eq1, k, v)
            set_color_by_tex(eq2, k, v)

        grp = VGroup(eq1, eq2).arrange_in_grid(2, 1)
        self.add(grp)


class ExampleScene10(Scene):
    def construct(self):
        # TODO: This approach to highlighting \sum does not work right now.
        # It changes the shape of the rendered equation.
        t2cm1 = {r"\\sum": BLUE, "n": RED, "1": GREEN, "x": YELLOW}
        t2cm2 = {r"\sum": BLUE, "n": RED, "1": GREEN, "x": YELLOW}
        eq1 = MathTex(
            r"\sum^{n}_{1} x", substrings_to_isolate=list(t2cm1.keys())
        ).scale(1.3)
        eq2 = MathTex(
            r"\sum_{1}^{n} x", substrings_to_isolate=list(t2cm2.keys())
        ).scale(1.3)

        def set_color_by_tex(
            mathtex: MathTex, tex: str, color: ParsableManimColor
        ) -> None:
            for match in mathtex.matched_strings_and_ids:
                if match[0] == tex:
                    mathtex.id_to_vgroup_dict[match[1]].set_color(color)

        for k, v in t2cm1.items():
            set_color_by_tex(eq1, k, v)
        for k, v in t2cm2.items():
            set_color_by_tex(eq2, k, v)

        grp = VGroup(eq1, eq2).arrange_in_grid(2, 1)
        self.add(grp)

        # This workaround based on index_labels still work
        # labels = index_labels(eq2)
        # self.add(labels)
        # eq1[0].set_color(BLUE)
        # eq2[1].set_color(BLUE)


class ExampleScene11(Scene):
    def construct(self):
        t2cm = {"n": RED, "1": GREEN, "x": YELLOW}
        eq = MathTex(r"\sum_{1}^{n} x", tex_to_color_map=t2cm).scale(1.3)

        self.add(eq)


class ExampleScene12(Scene):
    def construct(self):
        eq = MathTex(r"\sum_{1}^{n} x", substrings_to_isolate=["1", "n", "x"]).scale(
            1.3
        )
        eq.set_color_by_tex("1", YELLOW)
        eq.set_color_by_tex("x", RED)
        eq.set_opacity_by_tex("n", 0.5)

        self.add(eq)


class ExampleScene13(Scene):
    def construct(self):
        matrix_elements = [[1, 2, 3]]
        row = 0
        column = 2
        matrix = Matrix(matrix_elements)
        print(matrix.get_columns()[column][row].tex_string)


# Get inspiration from
# https://docs.manim.community/en/stable/guides/using_text.html#text-with-latex
