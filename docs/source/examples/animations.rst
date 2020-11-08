Animations
==========

.. manim:: RotationUpdater2

    class RotationUpdater2(Scene):
        def construct(self):
            def updater_forth(mobj, dt):
                mobj.rotate_about_origin(dt)
            def updater_back(mobj, dt):
                mobj.rotate_about_origin(-dt)
            line_reference = Line(ORIGIN, LEFT).set_color(WHITE)
            line_moving = Line(ORIGIN, LEFT).set_color(YELLOW)
            line_moving.add_updater(updater_forth)
            self.add(line_reference, line_moving)
            self.wait(2)
            line_moving.remove_updater(updater_forth)
            line_moving.add_updater(updater_back)
            self.wait(2)
            line_moving.remove_updater(updater_back)
            self.wait(0.5)


.. manim:: PointWithTrace

    class PointWithTrace(Scene):
        def construct(self):
            path = VMobject()
            dot = Dot()
            path.set_points_as_corners([dot.get_center(), dot.get_center()])
            def update_path(path):
                previus_path = path.copy()
                previus_path.add_points_as_corners([dot.get_center()])
                path.become(previus_path)
            path.add_updater(update_path)
            self.add(path, dot)
            self.play(Rotating(dot, radians=PI, about_point=RIGHT, run_time=2))
            self.wait()
            self.play(dot.shift, UP)
            self.play(dot.shift, LEFT)
            self.wait()

