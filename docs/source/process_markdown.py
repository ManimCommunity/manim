from pathlib import Path
import sys
import re

name = "test_raw"
project_path = sys.path[1]
path_source = Path(project_path) / "docs/source/"
file_markdown = path_source / f"{name}.md"
text_original = file_markdown.read_text(encoding=None, errors=None)
all_scenes_from_makrdown = re.findall(r"```[a-z]*\n[\s\S]*?\n```",
                                      text_original)  # find regular expression to match ```python\n  text ```

all_scenes_processed = []
for scene in all_scenes_from_makrdown:
    scene = scene[10:-3]  # removes ```python\n and ```
    all_scenes_processed.append(scene)

pathname = Path(project_path) / f"docs/source/media/py_{name}"
pathname.mkdir(parents=True, exist_ok=True)
for i, text in enumerate(all_scenes_processed):
    filename = Path(project_path) / f"docs/source/media/py_{name}/file{i}.py"
    filename.touch()
    filename.write_text(str(text))

# creating the gifs:
file_names = [subp for subp in pathname.rglob('*') if (".py" == subp.suffix)]
import os

for file in file_names:
    os.system(f"manim  -l -s -c 'BLACK' " + str(file))

final_doc = "test.md"
new_file = path_source / final_doc
new_file.touch()

counter = 0
def replacement_function(match):
    global counter
    img_loc_path = Path(f"media/images/file{counter}/")

    img_loc = [subp for subp in img_loc_path.rglob("*")]
    img_loc = img_loc[0]
    image_string = f"```\n![]({img_loc})"
    counter = + 1
    return image_string


p = re.compile(r'```\n')
new_text = p.sub(replacement_function, text_original)
new_file.write_text(new_text)
