from manim import *
from tests.helpers.graphical_units import set_test_scene


class YourClassHere(Scene):
    def construct(self):
        circle = Circle()
        self.play(Animation(circle))


set_test_scene(YourClassHere, "<whereitbelongs>")
