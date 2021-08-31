from manim import *


class SmoothUpdaters(Scene):
    def construct(self):
        config.frame_rate = 10
        sq = Square()
        sq.add_updater(lambda s, dt: s.rotate(dt))
        s1 = Square(1, color=RED).add_updater(lambda s, dt: s.move_to(sq.get_vertices()[0]).rotate(-4*dt))
        sq.add(s1)
        text = always_redraw(lambda: Text(str(sq.updating_speed)).to_corner(UR))
        self.add(text, sq)
        # self.play(FadeIn(sq), run_time=2)
        self.wait(2)
        sq.suspend_updating(recursive=False, run_time=1)
        self.wait(2)
        # sq.resume_updating(recursive=False, run_time=1)
        # self.wait(2)
        # sq.suspend_updating(run_time=2)
        # self.wait(3)
        # sq.resume_updating(recursive=False, run_time=3)
        # self.wait(4)


class NewGetArea(Scene):

    def f(self, x):
        return np.sin(x ** 2)

    def construct(self):
        ax = Axes()
        graph = ax.get_graph(self.f, x_range=[*ax.x_range[:2], 0.05])
        v_old = ValueTracker(2)
        area_old = ax.get_area_with_riemann_rectangles(graph, [1, v_old.get_value()], opacity=.1, color=YELLOW)
        v_new = ValueTracker(2)
        area_new = ax.get_area(graph, [-v_new.get_value(), -1], opacity=.6, color=YELLOW)
        self.add(ax, graph, area_old, area_new)
        # self.wait()
        # area_old.add_updater(lambda a: a.become(ax.get_area_with_riemann_rectangles(graph, [1, v_old.get_value()], opacity=.1, color=YELLOW)))
        # self.play(v_old.animate.set_value(5), run_time=6)
        # area_old.clear_updaters()
        self.wait()
        area_new.add_updater(
            lambda a: a.become(ax.get_area(graph, [-v_new.get_value(), -1], opacity=.6, color=YELLOW)))
        self.play(v_new.animate.set_value(5), run_time=6)
        area_new.clear_updaters()
        self.wait()