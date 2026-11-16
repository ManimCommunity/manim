# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

project = "Manim"
copyright = f"2020-{datetime.now().year}, The Manim Community Dev Team"  # noqa: A001
author = "The Manim Community Dev Team"

sys.path.insert(0, str(Path(__file__).parents[2].resolve()))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "manim.utils.docbuild.manim_directive",
]

templates_path = ["_templates"]
exclude_patterns: list[Any] = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

# Autodoc
autosummary_generate = False
autodoc_default_options = {"members": True, "show-inheritance": True}
