from manim import *

class Proof(Scene):
    def construct(self):

        title = Text("The Sum of Two Consecutive Integers Is Odd")
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

        eq1 = MathTex(r"m<n")
        self.play(Write(eq1))
        self.wait(1)

        eq2 = MathTex(r"n=m+1")
        self.play(Write(eq2))
        self.wait(2)

        eq3 = MathTex(r"m+n=m+(m+1)")
        self.play(Write(eq3))
        self.wait(2)

        eq4 = MathTex(r"m+n=2m+1")
        self.play(Transform(eq3, eq4))
        self.wait(2)

        conclusion = MathTex(
            r"2m+1\text{ is odd}",
            color=YELLOW
        )
        self.play(Write(conclusion))
        self.wait(2)
