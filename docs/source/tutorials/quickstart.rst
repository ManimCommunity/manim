from manim import *

class SquarePerimeter(Scene):
    def construct(self):
        # 1. ساخت مربع
        square = Square(side_length=3, color=BLUE)
        title = Text("محاسبه محیط مربع", font_size=40).to_edge(UP)
        
        # 2. برچسب‌گذاری اضلاع
        labels = VGroup(
            Text("a").next_to(square, UP),
            Text("a").next_to(square, RIGHT),
            Text("a").next_to(square, DOWN),
            Text("a").next_to(square, LEFT)
        )

        self.play(Write(title))
        self.play(Create(square))
        self.play(Write(labels))
        self.wait(1)

        # 3. انیمیشن کشیدن دور محیط
        perimeter_trace = square.copy().set_color(YELLOW).set_stroke(width=8)
        
        # فرمول مرحله به مرحله
        formula1 = MathTex("P", "=", "a", "+", "a", "+", "a", "+", "a").to_edge(DOWN, buff=1.5)
        
        self.play(ShowCreationThenDestruction(perimeter_trace), run_time=4)
        self.play(Write(formula1))
        self.wait(1)

        # 4. ساده‌سازی فرمول
        formula2 = MathTex("P", "=", "4", "\\times", "a").to_edge(DOWN, buff=1.5)
        
        self.play(TransformMatchingTex(formula1, formula2))
        self.play(formula2.animate.set_color(YELLOW).scale(1.2))
        self.wait(2)
