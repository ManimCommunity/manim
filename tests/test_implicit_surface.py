from manim import *
from manim.mobject.three_d.implicit_surface import ImplicitSurface

class TestSurfaceScene(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=70*DEGREES, theta=45*DEGREES)
        surface = ImplicitSurface(
            lambda x, y, z, r=0.7: x**2 + y**2 - r**2,
            resolution=20,
            color=WHITE
        )
        self.add(surface)
        self.wait()
