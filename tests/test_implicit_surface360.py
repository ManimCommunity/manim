from manim import *
from manim.mobject.three_d.implicit_surface import ImplicitSurface

class TestSurface360(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=0)  

        surface = ImplicitSurface(
            lambda x, y, z: x**2 + y**2 + z**2 - 1,
            resolution=20,
            color=BLUE
        )
        self.add(surface)

        self.begin_ambient_camera_rotation(rate=PI/4)  
        self.wait(8)
        self.stop_ambient_camera_rotation()
