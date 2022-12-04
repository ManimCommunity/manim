from manim import *

class testRemove(Scene):
    def construct(self):    
        sample_list = [1,2,3,4,3,2,1]  
        num_bins=len(sample_list)
        hist_list = list(np.histogram(sample_list, bins=4, density=True))
        hist = BarChart(hist_list[0], bar_colors=[ORANGE],
                        axis_config={},
                        x_axis_config={"include_numbers": True},
                        y_axis_config={"include_numbers": True}
        )
        self.add(hist)
        self.wait()
        for i,obj in enumerate(hist.y_axis.numbers):
            self.play(obj.animate.shift((0.3*i+.3)*LEFT))
            self.remove(hist.y_axis.numbers[i])
        self.wait(2) 

class testTrue(Scene):
    def construct(self):    
        sample_list = [1,2,3,4,3,2,1]  
        num_bins=len(sample_list)
        hist_list = list(np.histogram(sample_list, bins=4, density=True))
        hist = BarChart(hist_list[0], bar_colors=[ORANGE],
                        axis_config={},
                        x_axis_config={"include_numbers": True},
                        y_axis_config={"include_numbers": True}
        )
        self.add(hist)
        self.wait()
        self.wait(2) 

class testFalse(Scene):
    def construct(self):    
        sample_list = [1,2,3,4,3,2,1]  
        num_bins=len(sample_list)
        hist_list = list(np.histogram(sample_list, bins=4, density=True))
        hist = BarChart(hist_list[0], bar_colors=[ORANGE],
                        axis_config={},
                        x_axis_config={"include_numbers": False},
                        y_axis_config={"include_numbers": False}
        )
        self.add(hist)
        self.wait()
        self.wait(2) 

class testDefault(Scene):
    def construct(self):    
        sample_list = [1,2,3,4,3,2,1]  
        num_bins=len(sample_list)
        hist_list = list(np.histogram(sample_list, bins=4, density=True))
        hist = BarChart(hist_list[0], bar_colors=[ORANGE],
                        axis_config={},
        )
        self.add(hist)
        self.wait()
        self.wait(2) 

class ex1(Scene):
    def construct(self):
        l0 = NumberLine(
            include_numbers=True,
        )
        self.add(l0)
        self.play(l0.numbers.animate.shift(LEFT))
        self.wait()

class ex2(Scene):
    def construct(self):    
        l0 = NumberLine(
            include_numbers=True,
        )
        self.add(l0)
        self.remove(l0.numbers) # ← Remove what is supposed to contain only the tex numbers.
        self.play(l0.numbers.animate.shift(LEFT))
        self.wait()