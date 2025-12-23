from __future__ import annotations

from manim import *


class Scene1(Scene):
    def construct(self):
        formula = MathTex(
            r"P(X=k)",
            r" = 0.5^k (1-0.5)^{12-k}",
        )
        formula.id_to_vgroup_dict["unique001"].set_color(RED)
        self.add(formula)
        # self.add(formula.id_to_vgroup_dict['unique001'])


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


class Scene4(Scene):
    def construct(self):
        formula = MathTex(
            r"a^2 + b^2 = c^2 + a^2",
        ).scale(1.3)

        self.add(formula)


class Scene4a(Scene):
    r"""
    One small issue here is that the power 2 that b is raised to
    is moved a bit upwards than in Scene4 and Scene4b.
    This is related to adding a pair of curly braces around the
    detected substrings in MathTex::_handle_match
        pre_string = "{" + rf"\special{{dvisvgm:raw <g id='unique{ssIdx:03d}ss'>}}"
        post_string = r"\special{dvisvgm:raw </g>}}"
    If the curly braces are not added, the issue disappears.
    """

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
    """
    At an earlier stage, I experimented with using a regular expression to
    locate substrings to isolate. I don't think this is actually needed.
    """

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


class Scene6(Scene):
    def construct(self):
        formula = MathTex(
            r"a^2 + b^2 = c^2 + d^2 - a^2",
            substrings_to_isolate=["a", "c", "d"],
        ).scale(1.3)

        formula.set_color_by_tex("c", YELLOW)
        formula.set_color_by_tex("a", RED)

        self.add(formula)


class Scene7(Scene):
    def construct(self):
        formula = MathTex(
            r"a^2 + b^2 = c^2 + d^2 - 2 a^2",
            substrings_to_isolate=["a", "c", "d"],
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

        for texstring, color in t2cm.items():
            eq1.set_color_by_tex(texstring, color)
            eq2.set_color_by_tex(texstring, color)

        grp = VGroup(eq1, eq2).arrange_in_grid(2, 1)
        self.add(grp)


class Scene10(Scene):
    """
    This scene show an example of when the current implementation
    of substrings_to_isolate fails, as it changes the layout of the
    rendered latex equation.
    """

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

        for texstring, color in t2cm1.items():
            eq1.set_color_by_tex(texstring, color)
        for texstring, color in t2cm2.items():
            eq2.set_color_by_tex(texstring, color)

        grp = VGroup(eq1, eq2).arrange_in_grid(2, 1)
        self.add(grp)


class Scene10a(Scene):
    """
    This scene shows a workaround to the issue in Scene10, based on
    index_labels, it works, but currently the levelling is a bit different
    the previous implementation of MathTex.
    """

    def construct(self):
        t2cm = {"n": RED, "1": GREEN, "x": YELLOW}
        eq1 = MathTex(r"\sum^{n}_{1} x", substrings_to_isolate=list(t2cm.keys())).scale(
            1.3
        )
        eq2 = MathTex(r"\sum_{1}^{n} x", substrings_to_isolate=list(t2cm.keys())).scale(
            1.3
        )

        for texstring, color in t2cm.items():
            eq1.set_color_by_tex(texstring, color)
            eq2.set_color_by_tex(texstring, color)

        grp = VGroup(eq1, eq2).arrange_in_grid(2, 1)
        self.add(grp)

        # This workaround based on index_labels still work
        # labels = index_labels(eq2[0])
        # self.add(labels)
        eq2[0][1].set_color(BLUE)


class Scene11(Scene):
    def construct(self):
        t2cm = {"n": RED, "1": GREEN, "x": YELLOW}
        eq = MathTex(r"\sum_{1}^{n} x", tex_to_color_map=t2cm).scale(1.3)

        self.add(eq)


class Scene12(Scene):
    def construct(self):
        eq = MathTex(
            r"\sum_{1}^{n} x", substrings_to_isolate=["1", "n", "x"], use_svg_cache=True
        ).scale(1.3)
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
        print("matrix.get_columns()[column][row].tex_string")
        print(matrix.get_columns()[column][row].tex_string)
        self.add(matrix)


class Scene14(Scene):
    def construct(self):
        start = MathTex("2", "A", r"\to", "B").shift(UP)
        end = MathTex("2", "B", r"\to", "A")

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

    Currently this fails to run as expected.
    When inspecting the generated svg file, the group "unique000" that was inserted in the
    .tex file is not present...
    https://dvisvgm.de/Manpage/#specials
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


class Scene16a(Scene):
    """
    Currently this fails to run as expected.
    When inspecting the generated svg file, the group "unique000" that was inserted in the
    .tex file is not present...
    KeyError: unique000

    By adding this to MathTex::_break_up_by_substrings
    new_submobjects.append(self.id_to_vgroup_dict['root'])
    the code still runs, even if the expected group was not found.
    """

    def construct(self):
        self.add(Tex(r"$\frac{a}{b}$", tex_environment="center"))


class Scene16b(Scene):
    """This works fine."""

    def construct(self):
        self.add(Tex(r"$\frac{a}{b}$", tex_environment=None))


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


class Scene17a(Scene):
    """This seems to work fine right now."""

    def construct(self):
        n = MathTex("n").shift(LEFT)
        denominator = MathTex(r"\frac{ 2 }{ ", "n", " + 1 }").shift(2 * UP)
        self.add(n)
        self.wait(2)
        self.play(TransformMatchingTex(n, denominator))
        self.wait(2)

        numerator = MathTex(r"\frac{", "n", " + 1 }{ 2 }").shift(2 * DOWN)
        self.play(TransformMatchingTex(n, numerator))
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


class Scene19a(Scene):
    """
    This seems to work fine now.
    I have split the input string manually.
    """

    def construct(self):
        val_a = 3
        val_b = 2
        color_a = "#f1f514"
        color_b = "#cf0492"
        tex_a = (
            MathTex(str(val_a).format(int), color=color_a).move_to([-1, 2, 0]).scale(2)
        )
        tex_b = (
            MathTex(str(val_b).format(int), color=color_b).move_to([1, 2, 0]).scale(2)
        )

        self.play(FadeIn(tex_a, tex_b))
        self.wait()

        form = MathTex(
            r"\hat{u}= \frac{  ",
            "3",
            r" \hat{i} + ",
            "2",
            r"\hat{j}}{ \sqrt{ ",
            "2",
            "^{2} + ",
            "3",
            "^{2} } } }",
        ).scale(1.25)

        idx_a = [
            int(i)
            for i, character in enumerate(form)
            if character.tex_string == tex_a[0].tex_string
        ]
        idx_b = [
            int(i)
            for i, character in enumerate(form)
            if character.tex_string == tex_b[0].tex_string
        ]

        print(idx_a)
        print(idx_b)

        get_pos_a = form[idx_a[0]].get_center()
        get_pos_b = form[idx_b[0]].get_center()
        a1_copy = []
        b1_copy = []

        # here I force colouring of two's and three's on the MathTex object
        for i in range(len(idx_a)):
            a1_copy += form[idx_a[i]].set_color(color_a)
        for i in range(len(idx_b)):
            b1_copy += form[idx_b[i]].set_color(color_b)

        self.play(FadeIn(form))
        self.play(tex_a.animate.move_to(get_pos_a).match_height(form[idx_a[0]]))
        self.play(tex_b.animate.move_to(get_pos_b).match_height(form[idx_b[0]]))
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
    r"""
    LaTeX compilation error when breaking up a MathTex string by subscripts
    https://github.com/ManimCommunity/manim/issues/1865

    This currently fails.
    It can be solved by adding a pair of curly braces around the
    detected substrings in MathTex::_handle_match
        pre_string = "{" + rf"\special{{dvisvgm:raw <g id='unique{ssIdx:03d}ss'>}}"
        post_string = r"\special{dvisvgm:raw </g>}}"
    But doing that triggers and issue in Scene4a
    """

    def construct(self):
        # reqeq2 = MathTex(
        #    r"Y_{ij} = \mu_i + \gamma_i X_{ij} + e_{ij}", substrings_to_isolate=["i"]
        # )
        reqeq2 = MathTex(
            r"Y_{ij} = \mu_{i} + \gamma_{i} X_{ij} + e_{ij}",
            substrings_to_isolate=["i"],
        )
        self.add(reqeq2)


class Scene21a(Scene):
    """
    LaTeX compilation error when breaking up a MathTex string by subscripts
    https://github.com/ManimCommunity/manim/issues/1865

    By inserting curly braces around the objects to isolate,
    the error vanishes.
    """

    def construct(self):
        reqeq2 = MathTex(
            r"Y_{ij} = \mu_{i} + \gamma_{i} X_{ij} + e_{ij}",
            substrings_to_isolate=["i"],
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


class Scene23(Scene):
    def construct(self):
        """Test that set_opacity_by_tex works correctly."""
        tex = MathTex("f(x) = y", substrings_to_isolate=["f(x)"])
        print(tex.matched_strings_and_ids)
        tex.set_opacity_by_tex("f(x)", 0.2, 0.5)
        self.add(tex)


class Scene24(Scene):
    def construct(self):
        exp1 = MathTex("a^2", "+", "b^2", "=", "c^2").shift(2 * UP)
        exp2 = MathTex("a^2", "=", "c^2", "-", "b^2")
        exp3 = MathTex("a", "=", r"\sqrt{", "c^2", "-", "b^2", "}").shift(2 * DOWN)
        self.add(exp1)
        self.wait(2)
        self.play(TransformMatchingTex(exp1, exp2), run_time=5)
        self.play(TransformMatchingTex(exp2, exp3), run_time=5)
        self.wait(2)


class Scene25(Scene):
    """test_tex_white_space_and_non_whitespace_args(using_opengl_renderer):"""

    def construct(self):
        """Check that correct number of submobjects are created per string when mixing characters with whitespace"""
        separator = ", \n . \t\t"
        str_part_1 = "Hello"
        str_part_2 = "world"
        str_part_3 = "It is"
        str_part_4 = "me!"
        tex = Tex(
            str_part_1, str_part_2, str_part_3, str_part_4, arg_separator=separator
        )
        assert len(tex) == 4
        print(len(tex[0]))
        print(tex[0])
        print(len("".join((str_part_1 + separator).split())))
        print("".join((str_part_1 + separator).split()))
        assert len(tex[0]) == len("".join((str_part_1 + separator).split()))


class Scene26(Scene):
    def construct(self):
        eq = MathTex("{{ a }} + {{ b }} = {{ c }}")
        self.add(eq)
        eq.set_color_by_tex("a", YELLOW)


# Get inspiration from
# https://docs.manim.community/en/stable/guides/using_text.html#text-with-latex
