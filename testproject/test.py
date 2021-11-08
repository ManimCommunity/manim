from manim import *


class TestScene(Scene):
    def construct(self):
        matrix_r = Matrix([["x"], ["y"]])
        matrix_M = Matrix([["M_0"], ["M_1"]])
        m0 = Matrix([["x"], ["y"]])
        equal_sign = MathTex("=")
        right_arrow = Arrow(start=LEFT, end=RIGHT)

        lambda_condition_eq = VGroup(
            MathTex(r"Mr = \lambda r"),
            right_arrow,
            matrix_M,
            matrix_r,
            equal_sign,
            MathTex(r"\lambda"),
            matrix_r,  # .copy(),
        ).arrange(RIGHT)

        self.add(
            lambda_condition_eq
        )  # checks if all passed elements are an instance of VMObject, adds them to submobjects
        self.wait()


class AddToVGroup(Scene):
    def construct(self):
        circle_red = Circle(color=RED)
        circle_red2 = Circle(color=RED)
        circle_red2.shift(LEFT)
        circle_green = Circle(color=GREEN)
        circle_blue = Circle(color=BLUE)
        circle_red.shift(LEFT)
        circle_blue.shift(RIGHT)
        gr = VGroup(circle_red, circle_green, circle_red2)
        gr2 = VGroup(circle_blue)  # Constructor uses add directly
        self.add(gr, gr2)
        self.wait()
        gr += gr2  # Add group to another
        self.play(
            gr.animate.shift(DOWN),
        )
        gr -= gr2  # Remove group
        self.play(  # Animate groups separately
            gr.animate.shift(LEFT),
            gr2.animate.shift(UP),
        )
        self.play(  # Animate groups without modification
            (gr + gr2).animate.shift(RIGHT)
        )
        self.play(  # Animate group without component
            (gr - circle_red).animate.shift(RIGHT)
        )
