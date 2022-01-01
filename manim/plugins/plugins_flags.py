"""
plugins_flags.py
------------

Plugin Managing Utility.
"""

import pkg_resources

from manim import console

__all__ = ["list_plugins"]


def get_plugins():
    return {
        entry_point.name: entry_point.load()
        for entry_point in pkg_resources.iter_entry_points("manim.plugins")
    }


def list_plugins():
    console.print("[green bold]Plugins:[/green bold]", justify="left")

    plugins = get_plugins()
    for plugin in plugins:
        console.print(f" • {plugin}")
