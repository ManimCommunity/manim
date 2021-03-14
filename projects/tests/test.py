from manim import *


class Tests(Scene):
    def add_to_back_problem(self):
        a = Mobject()
        b = Line()
        a.add(a)
        a.add_to_back(b, b)

        print(a.submobjects)

    def construct(self):
        self.add_to_back_problem()


class NextToUpdater(Scene):
    def construct(self):
        def dot_position(mobject):
            mobject.set_value(dot.get_center()[0])
            mobject.next_to(dot.get_center(), direction=dot.get_center(), buff=0.1)

        dot = Dot(RIGHT * 3)
        label = DecimalNumber()
        label.add_updater(dot_position)
        self.add(dot, label)

        self.play(
            Rotating(dot, about_point=ORIGIN, angle=TAU, run_time=TAU, rate_func=linear)
        )


class DtUpdater(Scene):
    def construct(self):
        line = Square()

        # Let the line rotate 90° per second
        line.add_updater(lambda mobject, dt: mobject.rotate(dt * 90 * DEGREES))
        self.add(line)
        self.wait(2)
