from __future__ import annotations

from manim import *


class Example(Scene):
    def construct(self):
        t = MathTex(r"\int_{a}^{b} f(x) \;dx = F(b) - F(a)")
        self.play(Write(t))
        self.wait()
        self.play(Unwrite(t))
        self.wait()


class Intro(ThreeDScene):
    def construct(self):
        v1 = Arrow3D([0, 0, 0], [1, 1, 0])

        a = Tex("hi").move_to(v1.get_end())
        a.add_updater(lambda m: m.move_to(v1.get_end()))

        self.add(a, v1)
        self.play(Rotate(v1))


with tempconfig({"quality": "low_quality", "preview": True}):
    Intro().render()
