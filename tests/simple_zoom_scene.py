from manim import *


class test_zoom_pan_to_center(MovingCameraScene):
    def construct(self):
        s1 = Square()
        s1.set_x(-10)
        s2 = Square()
        s2.set_x(10)

        self.add(s1, s2)
        self.play(self.camera.auto_zoom(self.mobjects))
