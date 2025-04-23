
import numpy as np
from manim import *

class ColorTest(Scene):
    def construct(self):
        line1 = Line(np.array([-1, 0, 0]), np.array([0, 1, 0]), stroke_width=20).set_color(color=["#7301FF", "#EDCD00"])
        line2 = Line(np.array([-1, 0, 0]), np.array([1, 0, 0]), stroke_width=20).set_color(color=["#7301FF", "#EDCD00"])
        line3 = Line(np.array([-1, 0, 0]), np.array([0, -1, 0]), stroke_width=20).set_color(color=["#7301FF", "#EDCD00"])
        line4 = Line(np.array([-1, 0, 0]), np.array([-1, 1, 0]), stroke_width=20).set_color(color=["#7301FF", "#EDCD00"])

        self.play(Create(line1), Create(line2), Create(line3), Create(line4))
        self.wait(2)
