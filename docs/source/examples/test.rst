Test
=================================


.. manim:: Example1
    :quality: medium

    class Example1(Scene):
        def construct(self):
            def my_updater(mobj,dt):
                mobj.shift(UP*0.01)
            circ_reference = Circle().set_color(WHITE)
            tri = Triangle().set_color(RED)
            self.add(circ_reference, tri)
            self.wait(0.5)
            tri.add_updater(my_updater)
            self.wait(2)

.. manim:: Example2
    :quality: medium

    class Example2(Scene):
        def construct(self):
            def my_updater(mobj):
                mobj.shift(UP*0.01)
            circ_reference = Circle().set_color(WHITE)
            tri = Triangle().set_color(RED)
            self.add(circ_reference, tri)
            self.wait(0.5)
            tri.add_updater(my_updater)
            self.wait(2)


.. manim:: Example3
    :quality: medium

    class Example3(Scene):
    def construct(self):
        v_tracker = ValueTracker(0)
        def my_updater(mobj):
            mobj.shift(UP*0.01)
        circ_reference = Circle().set_color(WHITE)
        tri = Triangle().set_color(RED)
        self.add(circ_reference, tri)
        self.wait(0.5)
        tri.add_updater(my_updater)
        self.play(v_tracker.increment_value, 1 , run_time=2)


