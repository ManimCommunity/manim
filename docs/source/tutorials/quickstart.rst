

class RectangularPyramid(Scene):
    def construct(self):
        # Base rectangle
        base = Rectangle(width=4, height=2)
        base.set_fill(BLUE, opacity=0.5)
        base.shift(DOWN)

        # Apex point
        apex = Dot(UP * 2)

        # Lines from base corners to apex
        corners = base.get_vertices()
        edges = VGroup(*[Line(corner, apex.get_center()) for corner in corners])

        # Labels
        base_label = Text("Base").scale(0.5).next_to(base, DOWN)
        apex_label = Text("Apex").scale(0.5).next_to(apex, UP)

        # Animation
        self.play(Create(base), Write(base_label))
        self.play(FadeIn(apex), Write(apex_label))
        self.play(Create(edges))

        self.wait()

        # Rotate view
        self.play(Rotate(VGroup(base, edges, apex), angle=PI/2))
        self.wait()

        # Volume formula
        formula = MathTex("V = \\frac{1}{3} lwh")
        formula.to_edge(UP)
        self.play(Write(formula))

        # Example
        example = MathTex("V = \\frac{1}{3}(8)(6)(9) = 144")
        example.next_to(formula, DOWN)
        self.play(Write(example))

        self.wait(2)
