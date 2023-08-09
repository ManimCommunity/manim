from manim import *


class T(Scene):
    def construct(self):
        self.play(Write(MathTex("x^n+1")))
        
if __name__ == "__main__":
    T().render()