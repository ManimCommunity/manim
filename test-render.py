from manim import Scene, Circle, tempconfig

class MyScene(Scene):
    def construct(self):
        self.add(Circle())

with tempconfig({"preview": True}):
    MyScene().render()
