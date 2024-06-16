from __future__ import annotations

from manim import Circle, Scene, tempconfig


class MyScene(Scene):
    def construct(self):
        self.add(Circle())


with tempconfig({"preview": True}):
    MyScene().render()
