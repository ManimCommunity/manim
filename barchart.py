from manim import *


class histo(Scene):
    def construct(self):
        sample_list = [1, 2, 3, 4, 3, 2, 1]
        num_bins = len(sample_list)
        hist_list = list(np.histogram(sample_list, bins=4, density=True))
        hist = BarChart(
            hist_list[0],
            bar_colors=[ORANGE],
            axis_config={},
            x_axis_config={"include_numbers": False},
            y_axis_config={"include_numbers": False},
        )
        self.add(hist)
        self.wait()
        for i, obj in enumerate(hist.y_axis.numbers):
            self.play(obj.animate.shift((0.3 * i + 0.3) * LEFT))
            self.remove(hist.y_axis.numbers[i])
        self.wait(2)


with tempconfig({"quality": "medium_quality", "preview": True}):
    scene = histo()
    scene.render()
