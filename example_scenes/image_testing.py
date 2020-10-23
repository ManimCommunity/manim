from manim import *


class TestingImage(Scene):
    def construct(self):
        im1 = ImageMobject("low640×351.jpg", resolution_of_final_video=1080).shift(
            4 * LEFT
        )
        im2 = ImageMobject("middle_1280×701.jpg", resolution_of_final_video=1080).shift(
            4 * RIGHT
        )
        self.add(im1, im2)
        self.wait(1)
