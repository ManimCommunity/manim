from manim import *


class SquareToCircle(Scene):
    def construct(self):
        square = Square()
        circle = Circle()
        self.play(Transform(square, circle))


class FunctionLikeTest(Scene):
    def construct(self):
        assert "FunctionLike" in globals()
        a = FunctionLike()
        self.play(FadeIn(a))


class WithAllTest(Scene):
    def construct(self):
        assert "WithAll" in globals()
        a = WithAll()
        self.play(FadeIn(a))
