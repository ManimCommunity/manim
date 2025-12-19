from __future__ import annotations

from manim import *


class Scene2(Scene):
    def construct(self):
        formula = MathTex(
            r"P(X=k) = 0.5^k (1-0.5)^{12-k}",
        ).scale(1.3)
        print(formula.id_to_vgroup_dict.keys())
        # formula.id_to_vgroup_dict['unique002'].set_color(RED)
        # formula.set_color_by_tex("k", ORANGE)
        self.add(formula)


class Scene3(Scene):
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


class Scene4a(Scene):
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


class Scene4b(Scene):
    def construct(self):
        formula = MathTex(
            r"a^2 + b^2 = c^2 + a^2",
            substrings_to_isolate=["c", "a"],
        ).scale(1.3)
        for k in formula.id_to_vgroup_dict:
            print(k)
        for key in formula.id_to_vgroup_dict:
            if key[-2:] == "ss":
                formula.id_to_vgroup_dict[key].set_color(GREEN)

        self.add(formula)


class Scene5(Scene):
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
# Scene6 fails with the following error
# KeyError: 'unique001ss'
# I think it is related to a caching issue, because the error vanishes
# when the scene is rendered by itself.
# uv run manim render MathTexExamples.py Scene6
class Scene6(Scene):
    def construct(self):
        formula = MathTex(
            r"a^2 + b^2 = c^2 + d^2 - a^2",
            substrings_to_isolate=["[acd]"],
        ).scale(1.3)

        formula.set_color_by_tex("c", ORANGE)
        formula.set_color_by_tex("a", RED)

        self.add(formula)


class Scene7(Scene):
    def construct(self):
        formula = MathTex(
            r"a^2 + b^2 = c^2 + d^2 - 2 a^2",
            substrings_to_isolate=["[acd]"],
        ).scale(1.3)

        formula.set_color_by_tex("c", GREEN)
        formula.set_color_by_tex("a", RED)

        self.add(formula)


class Scene8(Scene):
    """
    Example based on this issue:
    set_color_by_tex selects wrong substring in certain contexts
    https://github.com/ManimCommunity/manim/issues/3492
    """

    def construct(self):
        formula = MathTex(
            r"P(X=k) =",
            r"\binom{12}{k}",
            r"0.5^{k}",
            r"(1-0.5)^{12-k}",
            substrings_to_isolate=["k", "1", "12", "0.5"],
        ).scale(1.3)

        formula.set_color_by_tex("k", GREEN)
        formula.set_color_by_tex("12", RED)
        formula.set_color_by_tex("1", YELLOW)
        formula.set_color_by_tex("0.5", BLUE_D)
        self.add(formula)


class Scene9(Scene):
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


class Scene10(Scene):
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


class Scene11(Scene):
    def construct(self):
        t2cm = {"n": RED, "1": GREEN, "x": YELLOW}
        eq = MathTex(r"\sum_{1}^{n} x", tex_to_color_map=t2cm).scale(1.3)

        self.add(eq)


class Scene12(Scene):
    def construct(self):
        eq = MathTex(r"\sum_{1}^{n} x", substrings_to_isolate=["1", "n", "x"]).scale(
            1.3
        )
        eq.set_color_by_tex("1", YELLOW)
        eq.set_color_by_tex("x", RED)
        eq.set_opacity_by_tex("n", 0.5)

        self.add(eq)


class Scene13(Scene):
    def construct(self):
        matrix_elements = [[1, 2, 3]]
        row = 0
        column = 2
        matrix = Matrix(matrix_elements)
        print(matrix.get_columns()[column][row].tex_string)


class Scene14(Scene):
    """
    Triggers this exception
    Exception in TransformMatchingTex::get_mobject_key
    """

    def construct(self):
        start = MathTex("A", r"\to", "B")
        end = MathTex("B", r"\to", "A")

        self.add(start)
        self.play(TransformMatchingTex(start, end, fade_transform_mismatches=True))


class Scene15(Scene):
    """
    Example taken from
    TeX splitting can cause the last parts of an equation to not be displayed
    https://github.com/ManimCommunity/manim/issues/2970

    This example seems to work well.
    """

    def construct(self):
        template = TexTemplate()
        template.add_to_preamble(r"""\usepackage[english]{babel}
        \usepackage{csquotes}\usepackage{cancel}""")
        lpc_implies_polynomial = (
            MathTex(
                r"s[t] = a_1 s[t-1] + a_2 s[t-2] + a_3 s[t-3] + \dots\\",
                r"{{\Downarrow}}\\",
                r"\text{Polynomial function}",
                tex_template=template,
                tex_environment="gather*",
            )
            .scale(0.9)
            .shift(UP * 1.5)
        )
        lpc_implies_not_polynomial = (
            MathTex(
                r"s[t] = a_1 s[t-1] + a_2 s[t-2] + a_3 s[t-3] + \dots\\",
                r"\xcancel{ {{\Downarrow}} }\\",
                r"\text{Polynomial function}",
                tex_template=template,
                tex_environment="gather*",
            )
            .scale(0.9)
            .shift(DOWN * 1.5)
        )
        self.add(lpc_implies_not_polynomial, lpc_implies_polynomial)


class Scene16(Scene):
    """
    LaTeX rendering incorrectly
    https://github.com/ManimCommunity/manim/issues/2912

    Problem is still present.
    I think it is related to the fraction line being a line object in the
    svg and not an object defined by bezier curves.
    """

    def construct(self):
        preamble = r"""
        %\usepackage[mathrm=sym]{unicode-math}
        %\setmathfont{Fira Math}
        """

        template = TexTemplate(
            preamble=preamble, tex_compiler="lualatex", output_format=".pdf"
        )

        self.add(Tex(r"$\frac{a}{b}$", tex_template=template))


class Scene17(Scene):
    """
    Tex splitting issue with frac numerator
    https://github.com/ManimCommunity/manim/issues/2884

    Problem is not solved at the moment.

    I don't think it is possible to make this
    type of animation / transform with the current implementation, as it
    requires the expression to be able to compile for each segment.
    """

    def construct(self):
        n = MathTex("n").shift(LEFT)
        denominator = MathTex(r"\frac{ 2 }{ n + 1 }", substrings_to_isolate="n").shift(
            2 * UP
        )
        self.add(n)
        self.wait(2)
        self.play(TransformMatchingTex(n, denominator))
        self.wait(2)

        numerator = MathTex(r"\frac{ n + 1 }{ 2 }", substrings_to_isolate="n").shift(
            2 * DOWN
        )
        self.play(TransformMatchingTex(n, numerator))
        self.wait(2)

        # This approach works fine.
        fraction_right = MathTex(r"{ ", "n", r" + 1 } \over { 2 }").shift(2 * RIGHT)
        self.play(TransformMatchingTex(n, fraction_right))
        self.wait(2)


class Scene18(Scene):
    """
    Transforming to similar MathTex object distorts sometimes
    https://github.com/ManimCommunity/manim/issues/2544

    Seems to work ok.
    """

    def construct(self):
        var = "a"
        a1 = MathTex("2").shift(UP)
        a2 = MathTex(var)
        a3 = MathTex(var).shift(DOWN)

        self.play(Write(a1))
        self.wait()
        self.play(Transform(a1, a2))
        self.wait()
        self.play(Transform(a1, a3))
        self.wait()


class Scene19(Scene):
    """
    Manim's unexpected colouring behaviour under the radical sign
    https://github.com/ManimCommunity/manim/issues/1996

    The code from the issue does not run at the moment.
    AttributeError: VMobjectFromSVGPath object has no attribute 'tex_string'

    The code have been adjusted to be able to run.
    """

    def construct(self):
        val_a = 3
        val_b = 2
        color_a = "#0470cf"
        color_b = "#cf0492"
        tex_a = Tex(str(val_a), color=color_a).move_to([-1, 2, 0]).scale(2)
        tex_b = Tex(str(val_b), color=color_b).move_to([1, 2, 0]).scale(2)

        self.play(FadeIn(tex_a, tex_b))
        self.wait()

        form = MathTex(
            # unique000
            r"\hat{u}= \frac{ 3 \hat{i}+ 2\hat{j}}{\sqrt{ {",
            # unique001
            r"2",
            # unique002
            r"^{2}+",
            # unique003
            r"3",
            # unique004
            r"^{2} } } } } }",
            substrings_to_isolate=["2", "3"],
        ).scale(1.25)

        get_pos_a = form.id_to_vgroup_dict["unique003"].get_center()
        get_pos_b = form.id_to_vgroup_dict["unique001"].get_center()

        form.id_to_vgroup_dict["unique003"].set_color(color_a)
        form.id_to_vgroup_dict["unique001"].set_color(color_b)

        self.play(FadeIn(form))
        self.play(
            tex_a.animate.move_to(get_pos_a).match_height(
                form.id_to_vgroup_dict["unique003"]
            )
        )
        self.play(
            tex_b.animate.move_to(get_pos_b).match_height(
                form.id_to_vgroup_dict["unique001"]
            )
        )
        self.wait(1)


class Scene20(Scene):
    """
    LaTex Error in combination of MathTex and SurroundingRectangle
    https://github.com/ManimCommunity/manim/issues/1907

    The example seems to be working fine.
    """

    def construct(self):
        if False:  # try to write            ... = \frac { u'v - uv' } {v^2}
            # for boxes:                           └─3─┘ └─5─┘

            text = MathTex(
                r"\left( \frac{u}{v} \right)'",  # 0
                "=",  # 1
                r"\frac {",  # 2   ⇐ error-stop at opening brace
                "u' v",  # 3 *
                "-",  # 4
                "u v'",  # 5 *
                "} {v^2}",  # 6   ⇐ closing brace
            )  # ⇑                                   ⇐ is here!

        else:  # instead use unwanted   ... = { u'v - uv' } \frac {1} {v^2}
            # for boxes:                    └─3─┘ └─5─┘

            text = MathTex(
                r"\left( \frac{u}{v} \right)'",  # 0
                "=",  # 1
                r"\left( \, ",  # 2
                "u' v",  # 3 *
                "-",  # 4
                "u v'",  # 5 *
                r"\, \right) \frac{1}{v^2}",  # 6
            )

        box1 = SurroundingRectangle(text[3], buff=0.1)
        box2 = SurroundingRectangle(text[5], buff=0.1, color=RED)

        self.play(Write(text))
        self.play(Create(box1))
        self.play(TransformFromCopy(box1, box2))

        self.wait(5)


class Scene21(Scene):
    """
    LaTeX compilation error when breaking up a MathTex string by subscripts
    https://github.com/ManimCommunity/manim/issues/1865

    This seems to work well.
    """

    def construct(self):
        reqeq2 = MathTex(
            r"Y_{ij} = \mu_i + \gamma_i X_{ij} + e_{ij}", substrings_to_isolate=["i"]
        )
        self.add(reqeq2)


class Scene22(Scene):
    """
    uwezi on discord 2025-05-27 Kerning (tex vs text)
    https://discord.com/channels/581738731934056449/1376977419269050460/1376998448724967454

    In the thread it is explained that the tex_environment should
    be given without the opening curly brace.
    tex_environment="minipage}{20em}"
    The code below seems to work fine.
    """

    def construct(self):
        tex = Tex(
            r"""
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Donec rhoncus eros turpis, quis ullamcorper augue pretium eget.
Nullam hendrerit massa at mauris lacinia, eget rhoncus
enim vestibulum.
""",
            tex_environment="{minipage}{20em}",
        )
        self.add(tex)


# Get inspiration from
# https://docs.manim.community/en/stable/guides/using_text.html#text-with-latex
