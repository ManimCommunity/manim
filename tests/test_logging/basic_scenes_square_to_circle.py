from __future__ import annotations

from manim.animation.transform import Transform
from manim.mobject.geometry.arc import Circle
from manim.mobject.geometry.polygram import Square
from manim.scene.scene import Scene

# This module is used in the CLI tests in tests_CLi.py.


class SquareToCircle(Scene):
    def construct(self):
        self.play(Transform(Square(), Circle()))
