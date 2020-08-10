import configparser
from pathlib import Path
import sys

config = configparser.ConfigParser()
manim_project_path = sys.path[1]
path_of_config = Path(manim_project_path) / "manim/default.cfg"
config.read(path_of_config)
fw_config = {}
root_folder = str(config['custom_folders'].get("custom_root"))
for opt in ['video_dir', 'tex_dir', 'text_dir', 'output_file']:
    sub_folder = config['custom_folders'].get(opt)
    path = Path.home() / root_folder / sub_folder
    fw_config[opt] = str(path)
print(fw_config)