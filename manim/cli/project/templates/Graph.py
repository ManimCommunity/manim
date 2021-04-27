from manim import *


class Graph(GraphScene):
    def __init__(self, **kwargs):
        GraphScene.__init__(
            self,
            y_min=0,
            y_max=100,
            y_axis_config={"tick_frequency": 10},
            y_labeled_nums=np.arange(0, 100, 10),
            **kwargs
        )

    def construct(self):
        self.setup_axes()
        dot = Dot().move_to(self.coords_to_point(PI / 2, 20))
        func_graph = self.get_graph(lambda x: 20 * np.sin(x))
        self.add(dot, func_graph)
