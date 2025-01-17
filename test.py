from manim import *

class Test(Scene):
    def construct(self):
        box = Square().set_fill("#C93")  # #CC9933 (주황색)

        self.play(FadeIn(box))
        self.wait()
