from manim import *


class MobjectArrangeTest(Scene):
    def construct(self):
        square = Square(color=RED)
        square.set_fill(color=RED, opacity=1.0)
        circ = Circle(color=GREEN)
        circ.set_fill(color=GREEN, opacity=1.0)
        tri = Triangle(color=BLUE)
        tri.set_fill(color=BLUE, opacity=1.0)
        star = Star(color=YELLOW)
        star.set_fill(color=YELLOW, opacity=1.0)

        group = (
            VGroup(square, circ, tri, star)
            .set_x(0)
            .arrange(RIGHT, buff=2, reference=None)
        )

        self.add(group)

        self.play(group.animate.arrange(RIGHT, 0.5, False, reference=star))
        self.play(
            group.animate.arrange(reference=circ, direction=UP, center=True, buff=0.5)
        )
        self.play(group.animate.arrange(LEFT, 0.5, True, tri))
        self.play(
            group.animate.arrange(
                center=False, reference=None, buff=0.5, direction=DOWN
            )
        )
        self.play(group.animate.arrange(RIGHT, 0.5, True, None))
        self.play(group.animate.arrange(RIGHT, 2, True))

        self.wait()
