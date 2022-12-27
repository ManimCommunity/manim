from __future__ import annotations

from manim import *
from manim.animation.fading import FadeIn
from manim.animation.transform import Transform
from manim.mobject.geometry.arc import Circle
from manim.mobject.geometry.polygram import Square
from manim.scene.scene import Scene


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
