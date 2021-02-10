from manim import *
import pytest


class SquareToCircle(Scene):
    def construct(self):
        square = Square()
        circle = Circle()
        self.play(Transform(square, circle))


class SceneWithMultipleCalls(Scene):
    def construct(self):
        number = Integer(0)
        self.add(number)
        for i in range(10):
            self.play(Animation(Square()))


class SceneWithMultipleWaitCalls(Scene):
    def construct(self):
        self.play(ShowCreation(Square()))
        self.wait(1)
        self.play(ShowCreation(Square().shift(DOWN)))
        self.wait(1)
        self.play(ShowCreation(Square().shift(2 * DOWN)))
        self.wait(1)
        self.play(ShowCreation(Square().shift(3 * DOWN)))
        self.wait(1)


class NoAnimations(Scene):
    def construct(self):
        dot = Dot().set_color(GREEN)
        self.add(dot)
        self.wait(1)


class SceneWithStaticWait(Scene):
    def construct(self):
        self.add(Square())
        self.wait()


class SceneWithNonStaticWait(Scene):
    def construct(self):
        s = Square()
        # Non static wait are triggered by mobject with time based updaters.
        s.add_updater(lambda mob, dt: None)
        self.add(s)
        self.wait()
