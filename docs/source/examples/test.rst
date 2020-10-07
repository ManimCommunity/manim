Test
=================================


.. manim:: Updater1Example
    :quality: medium

    class Updater1Example(Scene):
        def construct(self):
            def my_rotation_updater(mobj,dt):
                mobj.rotate_about_origin(0.1)
            line_reference = Line(ORIGIN, LEFT).set_color(WHITE)
            line_moving = Line(ORIGIN, LEFT).set_color(BLUE)
            line_moving.add_updater(my_rotation_updater)
            self.add(line_reference, line_moving)
            self.wait(2)

.. manim:: Updater2Example
    :quality: medium

    class Updater2Example(Scene):
        def construct(self):
            def my_rotation_updater(mobj):
                mobj.rotate_about_origin(0.1)
            line_reference = Line(ORIGIN, LEFT).set_color(WHITE)
            line_moving = Line(ORIGIN, LEFT).set_color(BLUE)
            line_moving.add_updater(my_rotation_updater)
            self.add(line_reference, line_moving)
            self.wait(2)


.. manim:: Updater3Example
    :quality: medium

    class Updater3Example(Scene):
        def construct(self):
            v_tracker = ValueTracker(0)
            def my_rotation_updater(mobj):
                mobj.rotate_about_origin(0.1)
            line_reference = Line(ORIGIN, LEFT).set_color(WHITE)
            line_moving = Line(ORIGIN, LEFT).set_color(BLUE)
            line_moving.add_updater(my_rotation_updater)
            self.add(line_reference, line_moving)
            self.play(v_tracker.increment_value, 1 , run_time=2)