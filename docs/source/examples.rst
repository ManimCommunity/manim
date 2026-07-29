from manim import *

class DUSP6Animation(Scene):
    def construct(self):
        # Title
        title = Text("DUSP6: Molecular Mechanism", font_size=36).to_edge(UP)
        self.play(Write(title))

        # Molecules
        erk = Circle(radius=1.2, color=RED, fill_opacity=0.6).shift(LEFT * 2)
        erk_label = Text("Active ERK1/2", font_size=20).move_to(erk)

        dusp6 = Square(side_length=1.5, color=BLUE, fill_opacity=0.8).shift(RIGHT * 2)
        dusp6_label = Text("DUSP6 (Brake)", font_size=20).move_to(dusp6)

        self.play(FadeIn(erk), Write(erk_label))
        self.play(FadeIn(dusp6), Write(dusp6_label))
        self.wait(1)

        # Dephosphorylation Action
        action_text = Text("Dephosphorylation", color=YELLOW, font_size=24).next_to(title, DOWN)
        self.play(Write(action_text))

        # Move DUSP6 to touch ERK cleanly
        self.play(dusp6.animate.next_to(erk, RIGHT, buff=0.1))

        # Create new inactive label and position it precisely over the circle
        inactive_label = Text("Inactive ERK", font_size=20).move_to(erk)
        self.play(
            erk.animate.set_color(GREEN),
            Transform(erk_label, inactive_label)
        )
        self.wait(2)
