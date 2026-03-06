"""
Demo Visualization for Manim Community Edition
================================================
This scene showcases several key Manim features:
  - Text & LaTeX rendering
  - Geometric shape animations and transforms
  - Function graphing with axes
  - Color transitions and updaters
"""

from manim import *


class DemoVisualization(Scene):
    def construct(self):
        # ── 1. Title ────────────────────────────────────────────────────────────
        title = Text("Manim Demo", font_size=60, color=BLUE)
        subtitle = Text("Community Edition", font_size=30, color=GRAY)
        subtitle.next_to(title, DOWN, buff=0.3)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP * 0.3))
        self.wait(0.8)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ── 2. Shapes & transforms ──────────────────────────────────────────────
        section = Text("Shapes & Transforms", font_size=36, color=YELLOW)
        section.to_edge(UP)
        self.play(Write(section))

        square = Square(side_length=2, color=BLUE, fill_opacity=0.4)
        circle = Circle(radius=1.2, color=GREEN, fill_opacity=0.4)
        triangle = Triangle(color=RED, fill_opacity=0.4).scale(1.5)

        self.play(Create(square))
        self.wait(0.3)
        self.play(Transform(square, circle))
        self.wait(0.3)
        self.play(Transform(square, triangle))
        self.wait(0.3)

        # Spin and shrink away
        self.play(Rotate(square, angle=PI, run_time=0.8), square.animate.scale(0))
        self.play(FadeOut(section))

        # ── 3. LaTeX equation ───────────────────────────────────────────────────
        section2 = Text("Mathematical Equations", font_size=36, color=YELLOW)
        section2.to_edge(UP)
        self.play(Write(section2))

        eq1 = MathTex(r"e^{i\pi} + 1 = 0", font_size=72)
        self.play(Write(eq1), run_time=2)
        self.wait(0.8)

        eq2 = MathTex(
            r"\int_{-\infty}^{\infty} e^{-x^2}\, dx = \sqrt{\pi}",
            font_size=54,
        )
        self.play(Transform(eq1, eq2), run_time=1.5)
        self.wait(0.8)
        self.play(FadeOut(eq1), FadeOut(section2))

        # ── 4. Function graph ───────────────────────────────────────────────────
        section3 = Text("Function Graphing", font_size=36, color=YELLOW)
        section3.to_edge(UP)
        self.play(Write(section3))

        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-1.5, 1.5, 0.5],
            x_length=10,
            y_length=4,
            axis_config={"color": WHITE, "include_tip": True},
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y")

        sine_curve = axes.plot(lambda x: np.sin(x), color=BLUE, x_range=[-4, 4])
        cosine_curve = axes.plot(lambda x: np.cos(x), color=RED, x_range=[-4, 4])

        sine_label = axes.get_graph_label(sine_curve, label=r"\sin(x)", x_val=2.5)
        cosine_label = axes.get_graph_label(cosine_curve, label=r"\cos(x)", x_val=1.2)

        self.play(Create(axes), Write(axes_labels))
        self.play(Create(sine_curve), Write(sine_label), run_time=2)
        self.play(Create(cosine_curve), Write(cosine_label), run_time=2)
        self.wait(0.8)

        # Animate a dot along the sine curve
        tracker = ValueTracker(-4)
        dot = always_redraw(
            lambda: Dot(
                axes.c2p(tracker.get_value(), np.sin(tracker.get_value())),
                color=YELLOW,
            )
        )
        self.play(FadeIn(dot))
        self.play(tracker.animate.set_value(4), run_time=3, rate_func=linear)
        self.wait(0.5)

        self.play(
            FadeOut(axes),
            FadeOut(axes_labels),
            FadeOut(sine_curve),
            FadeOut(cosine_curve),
            FadeOut(sine_label),
            FadeOut(cosine_label),
            FadeOut(dot),
            FadeOut(section3),
        )

        # ── 5. Closing ──────────────────────────────────────────────────────────
        outro = Text("Made with Manim", font_size=48, color=BLUE_C)
        link = Text("manim.community", font_size=28, color=GRAY)
        link.next_to(outro, DOWN, buff=0.4)

        self.play(Write(outro))
        self.play(FadeIn(link, shift=UP * 0.3))
        self.wait(1.5)
        self.play(FadeOut(outro), FadeOut(link))
