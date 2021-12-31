from manim import *

class ManimBanner2(Scene):
    def construct(self):
        ban = ManimBanner()
        shapes = ban[0]
        em = ban[1]
        shapes.get_center()
        self.play(Create(ManimBanner()),run_time=1)
        #self.play(SpiralIn(shapes), FadeIn(em),run_time=1)
        self.wait()
