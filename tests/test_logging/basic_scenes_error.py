from manim import *

# This module is intended to raise an error.


class Error(Scene):
    def construct(self):
        raise Exception("An error has occurred")
