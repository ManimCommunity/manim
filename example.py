from manim import *


class AntiderivativeExample(Scene):
    def construct(self):
        ax = Axes()
        graph1 = ax.plot(
            lambda x: np.log(np.abs(x)),
            discontinuities=[0],
            dt=0.05,
            use_smoothing=False,
            color=RED,
        )
        ar = ax
        graph2 = ax.plot_derivative_graph(
            graph1,
            discontinuities=[0],
            dt=0.05,
            use_smoothing=False,
            color=BLUE,
        )
        graph3 = ax.plot(
            lambda x: x * np.log(np.abs(x)) - x,
            discontinuities=[0],
            dt=0.05,
            use_smoothing=False,
            color=GREEN,
        )
        self.add(ax, graph1, graph2, graph3)
        self.interactive_embed()
