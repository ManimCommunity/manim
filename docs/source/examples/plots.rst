Plotting with Manim
===================

Examples to illustrate the use of :class:`.GraphScene` in manim.

.. manim:: SinAndCosFunctionPlot
    :save_last_frame:

    class SinAndCosFunctionPlot(GraphScene):
        CONFIG = {
            "x_min": -10,
            "x_max": 10.3,
            "num_graph_anchor_points": 100,
            "y_min": -1.5,
            "y_max": 1.5,
            "graph_origin": ORIGIN,
            "function_color": RED,
            "axes_color": GREEN,
            "x_labeled_nums": range(-10, 12, 2),
        }

        def construct(self):
            self.setup_axes(animate=False)
            func_graph = self.get_graph(np.cos, self.function_color)
            func_graph2 = self.get_graph(np.sin)
            vert_line = self.get_vertical_line_to_graph(TAU, func_graph, color=YELLOW)
            graph_lab = self.get_graph_label(func_graph, label="\\cos(x)")
            graph_lab2 = self.get_graph_label(func_graph2, label="\\sin(x)",
                                x_val=-10, direction=UP / 2)
            two_pi = MathTex(r"x = 2 \pi")
            label_coord = self.input_to_graph_point(TAU, func_graph)
            two_pi.next_to(label_coord, RIGHT + UP)
            self.add(func_graph, func_graph2, vert_line, graph_lab, graph_lab2, two_pi)

.. manim:: GraphAreaPlot
    :save_last_frame:

    class GraphAreaPlot(GraphScene):
        CONFIG = {
            "x_min" : 0,
            "x_max" : 5,
            "y_min" : 0,
            "y_max" : 6,
            "y_tick_frequency" : 1,
            "x_tick_frequency" : 1,
            "x_labeled_nums" : [0,2,3]
        }
        def construct(self):
            self.setup_axes()
            curve1 = self.get_graph(lambda x: 4 * x - x ** 2, x_min=0, x_max=4)
            curve2 = self.get_graph(lambda x: 0.8 * x ** 2 - 3 * x + 4, x_min=0, x_max=4)
            line1 = self.get_vertical_line_to_graph(2, curve1, DashedLine, color=YELLOW)
            line2 = self.get_vertical_line_to_graph(3, curve1, DashedLine, color=YELLOW)
            area1 = self.get_area(curve1, 0.3, 0.6, dx_scaling=10, area_color=BLUE)
            area2 = self.get_area(curve2, 2, 3, bounded=curve1)
            self.add(curve1, curve2, line1, line2, area1, area2)

.. manim:: HeatDiagramPlot
    :save_last_frame:

    class HeatDiagramPlot(GraphScene):
        CONFIG = {
            "y_axis_label": r"T[$^\circ C$]",
            "x_axis_label": r"$\Delta Q$",
            "y_min": -8,
            "y_max": 30,
            "x_min": 0,
            "x_max": 40,
            "y_labeled_nums": np.arange(-5, 34, 5),
            "x_labeled_nums": np.arange(0, 40, 5),
        }

        def construct(self):
            data = [20, 0, 0, -5]
            x = [0, 8, 38, 39]
            self.setup_axes()
            dot_collection = VGroup()
            for time, val in enumerate(data):
                dot = Dot().move_to(self.coords_to_point(x[time], val))
                self.add(dot)
                dot_collection.add(dot)
            l1 = Line(dot_collection[0].get_center(), dot_collection[1].get_center())
            l2 = Line(dot_collection[1].get_center(), dot_collection[2].get_center())
            l3 = Line(dot_collection[2].get_center(), dot_collection[3].get_center())
            self.add(l1, l2, l3)

