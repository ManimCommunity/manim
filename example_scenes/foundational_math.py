from manim import *


class FoundationalMath(Scene):
    def construct(self):
        text = Tex("Foundational Math Concepts").scale(1.5)
        self.play(Write(text))
        self.wait()
        self.play(FadeOut(text))


class SetTheory(Scene):
    def construct(self):
        # Create sets A and B
        circle_a = Circle(radius=2, color=BLUE, fill_opacity=0.5).shift(LEFT)
        circle_b = Circle(radius=2, color=GREEN, fill_opacity=0.5).shift(RIGHT)

        text_a = Tex("A").move_to(circle_a.get_center())
        text_b = Tex("B").move_to(circle_b.get_center())

        self.play(Create(circle_a), Create(circle_b), Write(text_a), Write(text_b))
        self.wait()

        # Union of A and B
        union_text = Tex("A $\cup$ B").to_corner(UP)
        self.play(Write(union_text))
        union_region = (
            Intersection(circle_a, circle_b)
            .set_color(YELLOW)
            .set_fill(YELLOW, opacity=0.5)
        )
        union_region_a = (
            Difference(circle_a, circle_b)
            .set_color(YELLOW)
            .set_fill(YELLOW, opacity=0.5)
        )
        union_region_b = (
            Difference(circle_b, circle_a)
            .set_color(YELLOW)
            .set_fill(YELLOW, opacity=0.5)
        )
        self.play(FadeIn(union_region_a), FadeIn(union_region_b), FadeIn(union_region))
        self.wait()
        self.play(
            FadeOut(union_region_a),
            FadeOut(union_region_b),
            FadeOut(union_region),
            FadeOut(union_text),
        )

        # Intersection of A and B
        intersection_text = Tex("A $\cap$ B").to_corner(UP)
        self.play(Write(intersection_text))
        intersection_region = (
            Intersection(circle_a, circle_b)
            .set_color(YELLOW)
            .set_fill(YELLOW, opacity=0.5)
        )
        self.play(FadeIn(intersection_region))
        self.wait()
        self.play(FadeOut(intersection_region), FadeOut(intersection_text))

        # Complement of A
        complement_text = Tex("A$^c$").to_corner(UP)
        self.play(Write(complement_text))
        complement_region = (
            Difference(
                Rectangle(width=10, height=10, color=WHITE).move_to(ORIGIN), circle_a
            )
            .set_color(YELLOW)
            .set_fill(YELLOW, opacity=0.5)
        )
        self.play(FadeIn(complement_region))
        self.wait()
        self.play(
            FadeOut(complement_region),
            FadeOut(complement_text),
            FadeOut(circle_a),
            FadeOut(circle_b),
            FadeOut(text_a),
            FadeOut(text_b),
        )


class FunctionGraphs(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            x_length=8,
            y_length=6,
            axis_config={"include_numbers": True},
        )
        self.play(Create(axes))

        # Linear function
        def linear_function(x):
            return 0.5 * x + 1

        linear_graph = axes.plot(linear_function, color=BLUE)
        linear_label = axes.get_graph_label(linear_graph, "y = 0.5x + 1", direction=UR)
        self.play(Create(linear_graph), Write(linear_label))
        self.wait()

        # Quadratic function
        def quadratic_function(x):
            return 0.2 * x**2 - 1

        quadratic_graph = axes.plot(quadratic_function, color=GREEN)
        quadratic_label = axes.get_graph_label(
            quadratic_graph, "y = 0.2x^2 - 1", direction=UR
        )
        self.play(Create(quadratic_graph), Write(quadratic_label))
        self.wait()

        # Exponential function
        def exponential_function(x):
            return 2**x / 5

        exponential_graph = axes.plot(exponential_function, color=RED, x_range=[-3, 3])
        exponential_label = axes.get_graph_label(
            exponential_graph, "y = 2^x/5", direction=UR
        )
        self.play(Create(exponential_graph), Write(exponential_label))
        self.wait()

        self.play(
            FadeOut(axes),
            FadeOut(linear_graph),
            FadeOut(linear_label),
            FadeOut(quadratic_graph),
            FadeOut(quadratic_label),
            FadeOut(exponential_graph),
            FadeOut(exponential_label),
        )


class GeometricShapes(Scene):
    def construct(self):
        # Create shapes
        circle = Circle(radius=2, color=BLUE, fill_opacity=0.5).shift(LEFT * 3)
        triangle = Triangle(color=GREEN, fill_opacity=0.5).shift(UP * 2)
        rectangle = Rectangle(width=4, height=3, color=RED, fill_opacity=0.5).shift(
            RIGHT * 3 + DOWN * 1
        )

        self.play(Create(circle), Create(triangle), Create(rectangle))
        self.wait()

        # Display area
        circle_area = MathTex("Area = \pi r^2").next_to(circle, DOWN)
        triangle_area = MathTex("Area = \frac{1}{2} bh").next_to(triangle, DOWN)
        rectangle_area = MathTex("Area = lw").next_to(rectangle, DOWN)
        self.play(Write(circle_area), Write(triangle_area), Write(rectangle_area))
        self.wait()

        # Display perimeter
        circle_perimeter = MathTex("Perimeter = 2\pi r").next_to(circle_area, DOWN)
        triangle_perimeter = MathTex("Perimeter = a+b+c").next_to(triangle_area, DOWN)
        rectangle_perimeter = MathTex("Perimeter = 2(l+w)").next_to(
            rectangle_area, DOWN
        )
        self.play(
            Write(circle_perimeter),
            Write(triangle_perimeter),
            Write(rectangle_perimeter),
        )
        self.wait()

        self.play(
            FadeOut(circle),
            FadeOut(triangle),
            FadeOut(rectangle),
            FadeOut(circle_area),
            FadeOut(triangle_area),
            FadeOut(rectangle_area),
            FadeOut(circle_perimeter),
            FadeOut(triangle_perimeter),
            FadeOut(rectangle_perimeter),
        )


class VectorVisualizations(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            x_length=8,
            y_length=6,
            axis_config={"include_numbers": True},
        )
        self.play(Create(axes))

        # Create vectors
        vector_a = Vector([2, 2], color=BLUE).shift(DOWN)
        vector_b = Vector([-1, 3], color=GREEN).shift(RIGHT)
        self.play(Create(vector_a), Create(vector_b))
        self.wait()

        # Vector addition
        vector_sum = Vector([1, 5], color=YELLOW).shift(DOWN + RIGHT)
        sum_label = MathTex("a+b").next_to(vector_sum, UR)
        self.play(Create(vector_sum), Write(sum_label))
        self.wait()

        # Scalar multiplication
        vector_scaled = Vector([4, 4], color=RED).shift(DOWN * 2)
        scaled_label = MathTex("2a").next_to(vector_scaled, UR)
        self.play(Transform(vector_a, vector_scaled), Write(scaled_label))
        self.wait()

        self.play(
            FadeOut(axes),
            FadeOut(vector_a),
            FadeOut(vector_b),
            FadeOut(vector_sum),
            FadeOut(sum_label),
            FadeOut(vector_scaled),
            FadeOut(scaled_label),
        )


class ProbabilityDistributions(Scene):
    def construct(self):
        # Uniform distribution
        axes_uniform = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 1, 0.2],
            x_length=8,
            y_length=4,
            axis_config={"include_numbers": True},
        ).shift(UP * 2)
        self.play(Create(axes_uniform))

        bars = []
        for i in range(10):
            bar = Rectangle(
                width=0.8, height=0.8, color=BLUE, fill_opacity=0.5
            ).move_to(axes_uniform.c2p(i + 0.5, 0.4))
            bars.append(bar)
            self.play(Create(bar), run_time=0.1)
        self.wait()

        # Normal distribution
        axes_normal = Axes(
            x_range=[-5, 5, 1],
            y_range=[0, 0.5, 0.1],
            x_length=8,
            y_length=4,
            axis_config={"include_numbers": True},
        ).shift(DOWN * 2)
        self.play(Create(axes_normal))

        def normal_distribution(x):
            return 0.4 * np.exp(-(x**2) / 2)

        normal_curve = axes_normal.plot(normal_distribution, color=RED)
        self.play(Create(normal_curve))
        self.wait()

        self.play(
            FadeOut(axes_uniform),
            *[FadeOut(bar) for bar in bars],
            FadeOut(axes_normal),
            FadeOut(normal_curve),
        )
