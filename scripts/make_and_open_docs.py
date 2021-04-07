import os
import sys
import webbrowser
from pathlib import Path

path_makefile = Path(__file__).parents[1] / "docs"
os.system(f"cd {path_makefile} && make html")

website = (path_makefile / "build" / "html" / "index.html").absolute().as_uri()
try:  # Allows you to pass a custom browser if you want.
    webbrowser.get(sys.argv[1]).open_new_tab(f"{website}")
except IndexError:
    webbrowser.open_new_tab(f"{website}")
