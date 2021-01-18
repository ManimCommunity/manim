from manim import *
class CustomFont(Scene):
    def construct(self):
        with register_font("Orbitron-VariableFont_wght.ttf"):
            a=CairoText("Hello",font="Orbitron")
        self.play(Write(a))