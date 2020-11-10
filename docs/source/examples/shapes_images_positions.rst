Shapes, Images and Positions
============================

.. manim:: PointMovingOnShapes

    class PointMovingOnShapes(Scene):
        def construct(self):
            circle = Circle(radius=1, color=BLUE)
            dot = Dot()
            dot2 = dot.copy().shift(RIGHT)
            self.add(dot)

            line = Line([3, 0, 0], [5, 0, 0])
            self.add(line)

            self.play(GrowFromCenter(circle))
            self.play(Transform(dot, dot2))
            self.play(MoveAlongPath(dot, circle), run_time=2, rate_func=linear)
            self.play(Rotating(dot, about_point=[2, 0, 0]), run_time=1.5)
            self.wait()


.. manim:: ManimCELogo
    :save_last_frame:

    class ManimCELogo(Scene):
        def construct(self):
            logo_green = "#87c2a5"
            logo_blue = "#525893"
            logo_red = "#e07a5f"
            ds_m = MathTex(r"\mathbb{M}", z_index=20).scale(7)
            ds_m.shift(2.25*LEFT + 1.5*UP)
            circle = Circle(color=logo_green,
                            fill_opacity=1,
                            z_index=7)
            square = Square(color=logo_blue,
                            fill_opacity=1,
                            z_index=5)
            triangle = Triangle(color=logo_red,
                                fill_opacity=1,
                                z_index=3)
            circle.shift(LEFT)
            square.shift(UP)
            triangle.shift(RIGHT)
            self.add(triangle, square, circle, ds_m) # Order matters
            self.wait()


.. manim:: GradientImageFromArray
    :save_last_frame:

    class GradientImageFromArray(Scene):
        def construct(self):
            n = 256
            imageArray = np.uint8(
                [[i * 256 / n for i in range(0, n)] for _ in range(0, n)]
            )
            image = ImageMobject(imageArray).scale(2)
            self.add(image)


.. manim:: MovingAround

    class MovingAround(Scene):
        def construct(self):
            square = Square(color=BLUE, fill_opacity=1)

            self.play(square.shift, LEFT)
            self.play(square.set_fill, ORANGE)
            self.play(square.scale, 0.3)
            self.play(square.rotate, 0.4)


.. manim:: BezierSpline
    :save_last_frame:

    class BezierSpline(Scene):
        def construct(self):
            np.random.seed(42)
            area = 4

            x1 = np.random.randint(-area, area)
            y1 = np.random.randint(-area, area)
            p1 = np.array([x1, y1, 0])
            destination_dot1 = Dot(point=p1).set_color(BLUE)

            x2 = np.random.randint(-area, area)
            y2 = np.random.randint(-area, area)
            p2 = np.array([x2, y2, 0])
            destination_dot2 = Dot(p2).set_color(RED)

            deltaP = p1 - p2
            deltaPNormalized = deltaP / get_norm(deltaP)

            theta = np.radians(90)
            r = np.array(
                (
                    (np.cos(theta), -np.sin(theta), 0),
                    (np.sin(theta), np.cos(theta), 0),
                    (0, 0, 0),
                )
            )
            senk = r.dot(deltaPNormalized)
            offset = 0.1
            offset_along = 0.5
            offset_connect = 0.25

            dest_line1_point1 = p1 + senk * offset - deltaPNormalized * offset_along
            dest_line1_point2 = p2 + senk * offset + deltaPNormalized * offset_along
            dest_line2_point1 = p1 - senk * offset - deltaPNormalized * offset_along
            dest_line2_point2 = p2 - senk * offset + deltaPNormalized * offset_along
            s1 = p1 - offset_connect * deltaPNormalized
            s2 = p2 + offset_connect * deltaPNormalized
            dest_line1 = Line(dest_line1_point1, dest_line1_point2)
            dest_line2 = Line(dest_line2_point1, dest_line2_point2)

            Lp1s1 = Line(p1, s1)

            Lp1s1.add_cubic_bezier_curve(
                s1,
                s1 - deltaPNormalized * 0.1,
                dest_line2_point1 + deltaPNormalized * 0.1,
                dest_line2_point1 - deltaPNormalized * 0.01,
            )
            Lp1s1.add_cubic_bezier_curve(
                s1,
                s1 - deltaPNormalized * 0.1,
                dest_line1_point1 + deltaPNormalized * 0.1,
                dest_line1_point1,
            )

            Lp2s2 = Line(p2, s2)

            Lp2s2.add_cubic_bezier_curve(
                s2,
                s2 + deltaPNormalized * 0.1,
                dest_line2_point2 - deltaPNormalized * 0.1,
                dest_line2_point2,
            )
            Lp2s2.add_cubic_bezier_curve(
                s2,
                s2 + deltaPNormalized * 0.1,
                dest_line1_point2 - deltaPNormalized * 0.1,
                dest_line1_point2,
            )

            mobjects = VGroup(
                Lp1s1, Lp2s2, dest_line1, dest_line2, destination_dot1, destination_dot2
            )

            mobjects.scale(2)
            self.add(mobjects)
