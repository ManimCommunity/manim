Annotations
===========

.. manim:: BraceAnnotation
    :save_last_frame:

    class BraceAnnotation(Scene):
        def construct(self):
            dot = Dot([-2, -1, 0])
            dot2 = Dot([2, 1, 0])
            line = Line(dot.get_center(), dot2.get_center()).set_color(ORANGE)
            b1 = Brace(line)
            b1text = b1.get_text("Horizontal distance")
            b2 = Brace(line, direction=line.copy().rotate(PI / 2).get_unit_vector())
            b2text = b2.get_tex("x-x_1")
            self.add(line, dot, dot2, b1, b2, b1text, b2text)

.. manim:: VectorArrow
    :save_last_frame:

    class VectorArrow(Scene):
        def construct(self):
            dot = Dot(ORIGIN)
            arrow = Arrow(ORIGIN, [2, 2, 0], buff=0)
            numberplane = NumberPlane()
            origin_text = Text('(0, 0)').next_to(dot, DOWN)
            tip_text = Text('(2, 2)').next_to(arrow.get_end(), RIGHT)
            self.add(numberplane, dot, arrow, origin_text, tip_text)

