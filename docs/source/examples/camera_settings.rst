Camera Settings
=================================

.. manim:: Example1
    :quality: medium

    class Example1(MovingCameraScene):
        def construct(self):
            text = Text("Hello World")
            self.add(text)
            self.play(self.camera_frame.set_width, text.get_width() * 1.2)
            self.wait()

.. manim:: Example2a
    :quality: medium

    class Example2a(MovingCameraScene):
        def construct(self):
            text = Text("Hello World").set_color(BLUE)
            self.add(text)
            self.camera_frame.save_state()
            self.play(self.camera_frame.set_width, text.get_width() * 1.2)
            self.wait(0.3)
            self.play(Restore(self.camera_frame))

.. manim:: Example2b
    :quality: medium

    class Example2b(MovingCameraScene):
        def construct(self):
            text = Text("Hello World").set_color(BLUE)
            self.add(text)
            self.play(self.camera_frame.set_width, text.get_width() * 1.2)
            self.wait(0.3)
            self.play(self.camera_frame.set_width, 14)


.. manim:: Example3
    :quality: medium

    class Example3(MovingCameraScene):
        def construct(self):
            s = Square(color=RED, fill_opacity=0.5).move_to(2 * LEFT)
            t = Triangle(color=GREEN, fill_opacity=0.5).move_to(2 * RIGHT)
            self.add(s, t)
            self.play(self.camera_frame.move_to, s)
            self.wait(0.3)
            self.play(self.camera_frame.move_to, t)

.. manim:: Example4
    :quality: medium

    class Example4(MovingCameraScene):
        def construct(self):
            s = Square(color=BLUE, fill_opacity=0.5).move_to(2 * LEFT)
            t = Triangle(color=YELLOW, fill_opacity=0.5).move_to(2 * RIGHT)
            self.add(s, t)
            self.play(self.camera_frame.move_to, s,
                      self.camera_frame.set_width,s.get_width()*2)
            self.wait(0.3)
            self.play(self.camera_frame.move_to, t,
                      self.camera_frame.set_width,t.get_width()*2)

            self.play(self.camera_frame.move_to, ORIGIN,
                      self.camera_frame.set_width,14)

.. manim:: Example5
    :quality: medium

    class Example5(GraphScene, MovingCameraScene):
        def setup(self):
            GraphScene.setup(self)
            MovingCameraScene.setup(self)
        def construct(self):
            self.camera_frame.save_state()
            self.setup_axes(animate=False)
            graph = self.get_graph(lambda x: np.sin(x),
                                   color=WHITE,
                                   x_min=0,
                                   x_max=3 * PI
                                   )
            dot_at_start_graph = Dot().move_to(graph.points[0])
            dot_at_end_grap = Dot().move_to(graph.points[-1])
            self.add(graph, dot_at_end_grap, dot_at_start_graph)
            self.play(self.camera_frame.scale, 0.5, self.camera_frame.move_to, dot_at_start_graph)
            self.play(self.camera_frame.move_to, dot_at_end_grap)
            self.play(Restore(self.camera_frame))
            self.wait()

.. manim:: Example6
    :quality: medium

    class Example6(GraphScene, MovingCameraScene):
        def setup(self):
            GraphScene.setup(self)
            MovingCameraScene.setup(self)
        def construct(self):
            self.camera_frame.save_state()
            self.setup_axes(animate=False)
            graph = self.get_graph(lambda x: np.sin(x),
                                   color=BLUE,
                                   x_min=0,
                                   x_max=3 * PI
                                   )
            moving_dot = Dot().move_to(graph.points[0]).set_color(ORANGE)

            dot_at_start_graph = Dot().move_to(graph.points[0])
            dot_at_end_grap = Dot().move_to(graph.points[-1])
            self.add(graph, dot_at_end_grap, dot_at_start_graph, moving_dot)
            self.play( self.camera_frame.scale,0.5,self.camera_frame.move_to,moving_dot)

            def update_curve(mob):
                mob.move_to(moving_dot.get_center())

            self.camera_frame.add_updater(update_curve)
            self.play(MoveAlongPath(moving_dot, graph, rate_func=linear))
            self.camera_frame.remove_updater(update_curve)

            self.play(Restore(self.camera_frame))