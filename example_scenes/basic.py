from manim import *


class SoundExample(Scene):
                # Source of sound under Creative Commons 0 License. https://freesound.org/people/Druminfected/sounds/250551/
                def construct(self):
                    dot = Dot().set_color(GREEN)
                    self.add_click_sound()
                    self.add(dot)
                    self.wait()
                   # self.add_sound("click.mp3")
                    dot.set_color(BLUE)
                    self.wait()
                  #  self.add_sound("click.mp3")
                    dot.set_color(RED)
                    self.wait()
import os ; import sys
from pathlib import Path
if __name__ == "__main__":
    project_path = Path(sys.path[1]).parent
    script_name = f"{Path(__file__).resolve()}"
    os.system(f"manim  -l --custom_folders  --disable_caching  -p -c 'BLACK' " + script_name)