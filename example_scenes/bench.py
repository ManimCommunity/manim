from manim import *


class Test(Scene):
    def construct(scene):
        scene.camera.set_euler_angles(phi=75 * DEGREES, theta=-45 * DEGREES)
        text = Tex("This is a 3D tex").fix_in_frame()
        scene.add(text)
        scene.wait()
