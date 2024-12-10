from manim import *

class TestAddUpdater(Scene):
    def construct(self):
        # Test 1: Simple updater to move an object
        square = Square().set_color(RED)
        square.add_updater(lambda m, dt: m.shift(RIGHT * dt))
        self.add(square)
        self.wait(2)  # Square should move to the right

        # Test 2: Updater to change color dynamically
        circle = Circle().set_color(BLUE)
        circle.add_updater(lambda m, dt: m.set_color_by_gradient(RED, YELLOW))
        self.add(circle)
        self.wait(2)  # Circle should change color

        # Test 3: Removing an updater during runtime
        triangle = Triangle().set_color(GREEN)
        def grow(triangle, dt):
            triangle.scale(1 + dt * 0.1)
        triangle.add_updater(grow)
        self.add(triangle)
        self.wait(1)
        triangle.remove_updater(grow)  # Stop updating
        self.wait(1)  # Triangle should stop growing

        # Test 4: Multiple updaters on a single object
        hexagon = RegularPolygon(n=6).set_color(PURPLE)
        hexagon.add_updater(lambda m, dt: m.rotate(dt))
        hexagon.add_updater(lambda m, dt: m.shift(UP * dt))
        self.add(hexagon)
        self.wait(2)  # Hexagon should rotate and move up

        # Test 5: Stress test with 10 objects and dynamic updates
        mobjects = [Square().shift(LEFT * i).set_color_by_gradient(RED, BLUE) for i in range(10)]
        for i, mobject in enumerate(mobjects):
            mobject.add_updater(lambda m, dt, i=i: m.shift(RIGHT * (dt + 0.01 * i)))
            self.add(mobject)
        self.wait(3)  # All objects should move right at different speeds
