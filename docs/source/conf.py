# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import manim
from manim.utils.docbuild.module_parsing import parse_module_attributes

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.


sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------

project = "Manim"
copyright = f"2020-{datetime.now().year}, The Manim Community Dev Team"  # noqa: A001
author = "The Manim Community Dev Team"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx_copybutton",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.viewcode",
    "sphinxext.opengraph",
    "manim.utils.docbuild.manim_directive",
    "manim.utils.docbuild.autocolor_directive",
    "manim.utils.docbuild.autoaliasattr_directive",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinxcontrib.programoutput",
    "myst_parser",
    "sphinx_design",
    "sphinx_reredirects",
]

# Automatically generate stub pages when using the .. autosummary directive
autosummary_generate = True

myst_enable_extensions = ["colon_fence", "amsmath"]

# redirects (for moved / deleted pages)
redirects = {
    "installation/linux": "uv.html",
    "installation/macos": "uv.html",
    "installation/windows": "uv.html",
}

# generate documentation from type hints
ALIAS_DOCS_DICT = parse_module_attributes()[0]
autodoc_typehints = "description"
autodoc_type_aliases = {
    alias_name: f"~manim.{module}.{alias_name}"
    for module, module_dict in ALIAS_DOCS_DICT.items()
    for category_dict in module_dict.values()
    for alias_name in category_dict
}
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

exclude_patterns: list[str] = []

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
    "source_repository": "https://github.com/ManimCommunity/manim/",
    "source_branch": "main",
    "source_directory": "docs/source/",
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

latex_engine = "lualatex"

# external links
extlinks = {
    "issue": ("https://github.com/ManimCommunity/manim/issues/%s", "#%s"),
    "pr": ("https://github.com/ManimCommunity/manim/pull/%s", "#%s"),
}

# opengraph settings
ogp_site_name = "Manim Community | Documentation"
ogp_site_url = "https://docs.manim.community/"
ogp_social_cards = {
    "image": "_static/logo.png",
}


# inheritance_graph settings
inheritance_graph_attrs = {
    "concentrate": True,
    "size": '""',
    "splines": "ortho",
    "nodesep": 0.1,
    "ranksep": 0.2,
}

inheritance_node_attrs = {
    "penwidth": 0,
    "shape": "box",
    "width": 0.05,
    "height": 0.05,
    "margin": 0.05,
}

inheritance_edge_attrs = {
    "penwidth": 1,
}

html_js_files = ["responsiveSvg.js"]

graphviz_output_format = "svg"
