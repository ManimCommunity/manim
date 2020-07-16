from manim import *

class Hello(Scene):
    def construct(self):
        dot = Dot()
        self.add(dot)
        self.wait(1)

from pathlib import Path
if __name__ == "__main__":
    script = f"{Path(__file__).resolve()}"
    os.system(f"manim  -l  -p --custom_folders   -c 'BLACK' " + script )