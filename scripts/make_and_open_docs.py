from __future__ import annotations

import subprocess
import sys
import webbrowser
from pathlib import Path

path_makefile = Path(__file__).resolve().parents[1] / "docs"
subprocess.run(["make", "html"], cwd=path_makefile)

website = (path_makefile / "build" / "html" / "index.html").absolute().as_uri()
try:  # Allows you to pass a custom browser if you want.
    webbrowser.get(sys.argv[1]).open_new_tab(f"{website}")
except IndexError:
    webbrowser.open_new_tab(f"{website}")
