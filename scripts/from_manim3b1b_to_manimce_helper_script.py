from pathlib import Path

example_dict = Path.cwd() / "manim_folder"
all_py_files = example_dict.rglob('*.py')

text_to_search = "from manimlib.imports import *"
replacement_text = "from manim import *"

for py_file in all_py_files:
    text = py_file.read_text()
    text = text.replace(text_to_search, replacement_text)
    py_file.write_text(text)
    print(py_file)

text_to_search = "TexMobject"
replacement_text = "MathTex"

for py_file in all_py_files:
    text = py_file.read_text()
    text = text.replace(text_to_search, replacement_text)
    py_file.write_text(text)




# further, the following magnetudes were removed, however you can still use them by adding this:
"""
FRAME_WIDTH = config["frame_width"]
FRAME_HEIGHT = config["frame_width"]

TOP = config["frame_height"] / 2 * UP
BOTTOM = config["frame_height"] / 2 * DOWN
LEFT_SIDE = config["frame_width"] / 2 * LEFT
RIGHT_SIDE = config["frame_width"] / 2 * RIGHT
"""