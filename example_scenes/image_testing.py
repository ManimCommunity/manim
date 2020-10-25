from manim import *


class TestingImage(Scene):
    def construct(self):
        im1 = ImageMobject("low640×351.jpg", scale_to_resolution=1080).shift(
            4 * LEFT
        )
        im2 = ImageMobject("middle_1280×701.jpg", scale_to_resolution=1080).shift(
            4 * RIGHT
        )
        self.add(im1, im2)
        self.wait(1)

import os ; import sys
from pathlib import Path 
if __name__ == "__main__":
    project_path = Path(sys.path[1]).parent
    script_name = f"{Path(__file__).resolve()}"
    os.system(f"manim  -l --custom_folders  --disable_caching -s -p -c 'BLACK' --config_file '{project_path}/manim_settings.cfg' " + script_name)