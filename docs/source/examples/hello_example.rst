SomeTestExample
=================================

.. manim:: DotScene1
    :quality: low
    :save_last_frame:

    class DotScene1(Scene):
       def construct(self):
           dot = Dot().set_color(GREEN)
           self.add(dot)
           self.wait(1)

.. manim:: DotScene2
    :quality: medium
    :save_last_frame:

    class DotScene2(Scene):
       def construct(self):
           dot = Dot().set_color(YELLOW)
           self.add(dot)
           self.wait(1)

.. manim:: DotScene3
    :quality: high
    :save_last_frame:

    class DotScene3(Scene):
       def construct(self):
           dot = Dot().set_color(RED)
           self.add(dot)
           self.wait(1)

.. manim:: DotScene4
    :quality: high

    class DotScene4(Scene):
        def construct(self):
            dot = Dot().set_color(YELLOW).scale(3)
            self.add(dot)
            sq = Square()
            self.play(Transform(dot,sq))
            self.wait(1)

