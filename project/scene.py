from manim import *

class FirstSquare (Scene):
    def construct(self):
       # animate square 1
       square = Square(stroke_width=50).move_to(LEFT * 2)
       self.play(Create(square))
       self.wait()

class SecondSquare (Scene):
    def construct(self):
      # animate square 2
      rounded_Square = RoundedRectangle(corner_radius=0.2,stroke_width=50,height=2.1,width=2.1).move_to(RIGHT*2)
      self.play(Create(rounded_Square))
      self.wait()
        

class TwoSquares(Scene):
    def construct(self):
        scenes = [FirstSquare,SecondSquare]
        FirstSquare.construct(self)
        SecondSquare.construct(self)
        for scene in scenes:
            scene.construct(self)