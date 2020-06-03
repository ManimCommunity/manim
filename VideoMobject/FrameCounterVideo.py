from manim import *

class VideoTest(Scene):
    def construct(self):
        def func1(num):
            text =Text(f"{num}",font='Arial').scale(3)
            text.num= num
            return text
        text= func1(0)
        def update_curve(d,dt):
            d.num = d.num+1
            d.become(func1(d.num))
        self.add(text)
        text.add_updater(update_curve)
        self.wait(3)

from pathlib import Path
if __name__ == "__main__":
    script = f"{Path(__file__).resolve()}"
    os.system(f"manim  -l -p -c 'ORANGE' --video_dir ~/Downloads/ " + script + " VideoTest")