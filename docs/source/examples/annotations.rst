Annotations
=================================

.. manim:: AnnotateBrace
    :save_last_frame:

    class AnnotateBrace(Scene):
        def construct(self):
            dot = Dot([0, 0, 0])
            dot2 = Dot([2, 1, 0])
            line = Line(dot.get_center(), dot2.get_center()).set_color(ORANGE)
            b1 = Brace(line)
            b1text = b1.get_text("Distance")
            b2 = Brace(line, direction=line.copy().rotate(PI / 2).get_unit_vector())
            b2text = b2.get_tex("x-x_1")
            self.add(dot, dot2, line, b1, b2, b1text, b2text)

.. manim:: ExampleArrow
    :quality: medium
    :save_last_frame:

    class ExampleArrow(Scene):
        def construct(self):
            dot = Dot(ORIGIN)
            arrow = Arrow(ORIGIN, [2, 2, 0], buff=0)
            numberplane = NumberPlane()
            origin_text = TextMobject('(0, 0)').next_to(dot, DOWN)
            tip_text = TextMobject('(2, 2)').next_to(arrow.get_end(), RIGHT)
            self.add(numberplane, dot, arrow, origin_text, tip_text)

.. manim:: ExampleArrow2
    :quality: medium
    :save_last_frame:

    class ExampleArrow2(Scene):
        def construct(self):
            a11 = Arrow(np.array([-2, 3, 0]), np.array([2, 3, 0]))
            a12 = Arrow(np.array([-2, 2, 0]), np.array([2, 2, 0]),
                        tip_shape=ArrowTriangleTip,
                        tip_style={'fill_opacity': 0, 'stroke_width': 3})
            a21 = Arrow(np.array([-2, 1, 0]), np.array([2, 1, 0]), tip_shape=ArrowSquareTip)
            a22 = Arrow(np.array([-2, 0, 0]), np.array([2, 0, 0]), tip_shape=ArrowSquareTip,
                        tip_style={'fill_opacity': 0, 'stroke_width': 3})
            a31 = Arrow(np.array([-2, -1, 0]), np.array([2, -1, 0]), tip_shape=ArrowCircleTip)
            a32 = Arrow(np.array([-2, -2, 0]), np.array([2, -2, 0]), tip_shape=ArrowCircleTip,
                        tip_style={'fill_opacity': 0, 'stroke_width': 3})
            self.add(a11, a12, a21, a22, a31, a32)
            b11 = a11.copy().scale(0.5, scale_tips=True).next_to(a11, RIGHT)
            b12 = a12.copy().scale(0.5, scale_tips=True).next_to(a12, RIGHT)
            b21 = a21.copy().scale(0.5, scale_tips=True).next_to(a21, RIGHT)

            self.add(b11, b12, b21)
            self.wait(1)