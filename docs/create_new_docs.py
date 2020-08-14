from pathlib import Path
import os
import webbrowser

path_makefile= Path().home() / "projects/manim-community/docs"
os.system(f"cd {path_makefile} && make html")
website = Path().home() / "projects/manim-community/docs/build/html/test.html"
webbrowser.get('firefox').open_new_tab(f"{website}")