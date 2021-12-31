from manim import *

class ManimBanner2(Scene):
    def construct(self):
        ban = ManimBanner()
        shapes = ban[0]
        em = ban[1]
        shapes.get_center()
        self.play(WiggleIn(shapes),run_time=1)
        self.wait()
