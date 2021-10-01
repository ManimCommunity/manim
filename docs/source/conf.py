# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
from distutils.sysconfig import get_python_lib
from pathlib import Path

import manim

sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------

project = "Manim"
copyright = "2020-2021, The Manim Community Dev Team"
author = "The Manim Community Dev Team"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "recommonmark",
    "sphinx_copybutton",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.viewcode",
    "sphinxext.opengraph",
    "manim_directive",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
]

# Automatically generate stub pages when using the .. autosummary directive
autosummary_generate = True

# generate documentation from type hints
autodoc_typehints = "description"
autoclass_content = "both"

# controls whether functions documented by the autofunction directive
# appear with their full module names
add_module_names = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Custom section headings in our documentation
napoleon_custom_sections = ["Tests", ("Test", "Tests")]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
html_extra_path = ["robots.txt"]

exclude_patterns = []

# -- Options for internationalization ----------------------------------------
# Set the destination directory of the localized po files
locale_dirs = ["../i18n/"]

# Splits the text in more pot files.
gettext_compact = False

# Remove useless metadata from po files.
gettext_last_translator = ""
gettext_language_team = ""

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "furo"
html_favicon = str(Path("_static/favicon.ico"))

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {
    "light_logo": "manim-logo-sidebar.svg",
    "dark_logo": "manim-logo-sidebar-dark.svg",
    "light_css_variables": {
        "color-content-foreground": "#000000",
        "color-background-primary": "#ffffff",
        "color-background-border": "#ffffff",
        "color-sidebar-background": "#f8f9fb",
        "color-brand-content": "#1c00e3",
        "color-brand-primary": "#192bd0",
        "color-link": "#c93434",
        "color-link--hover": "#5b0000",
        "color-inline-code-background": "#f6f6f6;",
        "color-foreground-secondary": "#000",
    },
    "dark_css_variables": {
        "color-content-foreground": "#ffffffd9",
        "color-background-primary": "#131416",
        "color-background-border": "#303335",
        "color-sidebar-background": "#1a1c1e",
        "color-brand-content": "#2196f3",
        "color-brand-primary": "#007fff",
        "color-link": "#51ba86",
        "color-link--hover": "#9cefc6",
        "color-inline-code-background": "#262626",
        "color-foreground-secondary": "#ffffffd9",
    },
}
html_title = f"Manim Community v{manim.__version__}"

# This specifies any additional css files that will override the theme's
html_css_files = ["custom.css"]


# external links
extlinks = {
    "issue": ("https://github.com/ManimCommunity/manim/issues/%s", "issue "),
    "pr": ("https://github.com/ManimCommunity/manim/pull/%s", "pull request "),
}

# opengraph settings
ogp_image = "https://www.manim.community/logo.png"
ogp_site_name = "Manim Community | Documentation"
ogp_site_url = "https://docs.manim.community/"


# inheritance_graph settings
inheritance_graph_attrs = dict(
    concentrate=True,
    size='""',
    splines="ortho",
    nodesep=0.1,
    ranksep=0.2,
)

inheritance_node_attrs = dict(
    penwidth=0,
    shape="box",
    width=0.05,
    height=0.05,
    margin=0.05,
)

inheritance_edge_attrs = dict(
    penwidth=1,
)

html_js_files = [
    "responsiveSvg.js",
]

graphviz_output_format = "svg"
