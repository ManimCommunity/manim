from __future__ import annotations

from manim.scene.scene import Scene

# This module is intended to raise an error.


class Error(Scene):
    def construct(self):
        raise Exception("An error has occurred")
