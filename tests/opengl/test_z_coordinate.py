from manim import *

class TestZCoordinateSorting(Scene):
    def construct(self):
        # Test 1: Basic z-sorting with static Mobjects
        square = Square().shift(OUT).set_color(RED)
        circle = Circle().shift(IN).set_color(BLUE)
        self.add(square, circle)
        self.wait(1)

        # Test 2: Z-index sorting with multiple Mobjects
        triangle = Triangle().set_z_index(3).set_color(GREEN)
        square.set_z_index(1)
        circle.set_z_index(2)
        self.add(triangle)
        self.wait(1)

        # Test 3: Dynamic z-sorting during animations
        self.play(circle.animate.set_z_index(4), run_time=2)
        self.play(square.animate.set_z_index(5), run_time=2)
        self.wait(1)

        # Test 4: Overlapping objects with z-index
        hexagon = RegularPolygon(n=6).set_color(PURPLE).shift(OUT * 1.5).set_z_index(0)
        self.add(hexagon)
        self.wait(1)

        # Test 5: Stress test with 10 overlapping Mobjects
        for i in range(10):
            mobject = Circle(radius=0.2).shift(OUT * i * 0.2).set_z_index(i).set_color_by_gradient(RED, YELLOW)
            self.add(mobject)
        self.wait(2)

        # Test 6: Combined test with transformations and z-sorting
        pentagon = RegularPolygon(n=5).set_color(ORANGE).set_z_index(3)
        self.add(pentagon)
        self.play(pentagon.animate.shift(UP).set_z_index(6), run_time=2)
        self.wait(1)
